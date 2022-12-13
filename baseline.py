import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import time
import random

from transformers import DataCollatorForLanguageModeling

from trl.core import (
    logprobs_from_logits,
    whiten,
    clip_by_value,
    entropy_from_logits,
    flatten_dict,
    average_torch_dicts,
    stats_to_np,
    stack_dicts,
    add_suffix,
    WANDB_PADDING,
)

from trl.ppo import AdaptiveKLController, FixedKLController


class BaselineTrainer:
    def __init__(self, model, ref_model, tokenizer, **baseline_params):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.baseline_params = baseline_params
        self.optimizer = Adam(model.parameters(), lr=self.baseline_params["lr"])
        self.device = next(model.parameters()).device

        if self.baseline_params["adap_kl_ctrl"]:
            self.kl_ctl = AdaptiveKLController(
                self.baseline_params["init_kl_coef"],
                self.baseline_params["target"],
                self.baseline_params["horizon"],
            )
        else:
            self.kl_ctl = FixedKLController(self.baseline_params["init_kl_coef"])

    def step(
        self,
        queries: list[torch.Tensor],
        responses: list[torch.Tensor],
        scores: list[float],
    ):
        bs = self.ppo_params["batch_size"]
        assert bs == len(
            queries
        ), f"Batch size ({bs}) does not match number of examples ({len(queries)})"

        timing = dict()
        t0 = time.time()

        t = time.time()
        logprobs, ref_logprobs, values = self.batched_forward_pass(queries, responses)
        timing["time/ppo/forward_pass"] = time.time() - t

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing["time/ppo/compute_rewards"] = time.time() - t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        for _ in range(self.ppo_params["ppo_epochs"]):
            random.shuffle(idxs)
            for i in range(bs):
                idx = idxs[i]
                train_stats = self.train_minibatch(
                    logprobs[idx].unsqueeze(0),
                    rewards[idx].unsqueeze(0),
                )
                all_stats.append(train_stats)
        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(
            train_stats["policy/advantages"]
        ).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(
            train_stats["policy/advantages"], WANDB_PADDING
        )
        train_stats["policy/ratio"] = torch.flatten(
            train_stats["policy/ratio"]
        ).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
        )
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t

        self.kl_ctl.update(stats["objective/kl"], self.ppo_params["batch_size"])

        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)
        return stats

    def batched_forward_pass(
        self, queries: list[torch.Tensor], responses: list[torch.Tensor]
    ):
        """Calculate model outputs in multiple batches."""
        bs = self.ppo_params["batch_size"]
        fbs = self.ppo_params["forward_batch_size"]
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        for i in range(int(bs / fbs)):
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            input_ids = self.data_collator(
                [torch.cat([q, r]) for q, r in zip(query_batch, response_batch)]
            )["input_ids"]
            with torch.no_grad():
                logits, _, v = self.model(input_ids)
                ref_logits, _, _ = self.ref_model(input_ids)
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
            for j in range(fbs):
                start = len(query_batch[j]) - 1
                end = len(query_batch[j]) + len(response_batch[j]) - 1
                all_values.append(v[j, start - 1 : end - 1])
                all_logprobs.append(logprobs[j, start:end])
                all_ref_logprobs.append(ref_logprobs[j, start:end])
        return all_logprobs, all_ref_logprobs, all_values

    def train_minibatch(self, logprobs: torch.Tensor, rewards: list[float]):
        """Train one PPO minibatch"""
        loss, train_stats = self.loss(logprobs, rewards)
        # loss_p, loss_v, train_stats  = self.loss(logprobs, values, rewards, query, response, model_input)
        # loss = loss_p + loss_v
        self.optimizer.zero_grad()
        # print optimizer learning rate
        # print(f"learning rate: {self.optimizer.param_groups[0]['lr']}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        train_stats["grad_norm"] = grad_norm
        self.optimizer.step()
        return train_stats

    def compute_rewards(
        self,
        scores: list[float],
        logprobs: list[torch.Tensor],
        ref_logprobs: list[torch.Tensor],
    ):
        """Compute per token rewards from scores and KL-penalty."""
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            reward[-1] += score
            rewards.append(reward)
        return rewards, non_score_rewards

    def loss(self, old_logprobs: torch.Tensor, rewards: list[float]):
        """Calculate losses: negative log likelihood weighted by reward."""
        scaled_logprobs = old_logprobs.clone()
        for i in range(len(rewards)):
            if not rewards[i]:
                scaled_logprobs[i] *= -0.1
        loss = -scaled_logprobs.mean()
        stats = dict(loss=loss)
        return loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl_list = [
            logprobs - ref_logprobs
            for logprobs, ref_logprobs in zip(data["logprobs"], data["ref_logprobs"])
        ]
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        mean_entropy = torch.mean(
            torch.stack([torch.sum(-log_probs) for log_probs in data["logprobs"]])
        )
        mean_non_score_reward = torch.mean(
            torch.stack(
                [
                    torch.sum(non_score_reward)
                    for non_score_reward in data["non_score_reward"]
                ]
            )
        )
        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
        }

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = (
            1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        )
        return stats