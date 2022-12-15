# %%
# imports
import torch
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer
import wandb
from tqdm import tqdm
import time
from datasets import load_dataset
import numpy as np
import re
from datetime import datetime

from baseline import BaselineTrainer

# %%
config = {
    "steps": 200000,
    "batch_size": 256,
    "minibatch_size": 256,
    "forward_batch_size": 16,
    "txt_in_len": 16,
    "txt_out_len": 32,
    "lr": 1e-6,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": 0.2,
    "cliprange_value": 0.2,
    "vf_coef": 0.1,
    "ppo_epochs": 1,
}
# %%
# get models

model_name = "gpt2"  # "gpt2-medium"
gpt2_model = GPT2HeadWithValueModel.from_pretrained(model_name)
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)
gpt2_model_ref.to(device)

gen_kwargs = {
    "min_length": -1,
    "top_k": 50,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
}

# %%
# load imdb with datasets
ds = load_dataset("imdb", split="train")
ds.rename_column_("text", "review")
ds.rename_column_("label", "sentiment")
ds = ds.filter(lambda x: len(x["review"]) > 200, batch_size=None)

# %%
# randomize the query and response lengths
# class LengthSampler:
#     def __init__(self, min_value, max_value):
#         self.values = list(range(min_value, max_value))

#     def __call__(self):
#         return np.random.choice(self.values)


# input_size = LengthSampler(config["txt_in_min_len"], config["txt_in_max_len"])
# output_size = LengthSampler(config["txt_out_min_len"], config["txt_out_max_len"])

input_len = config["txt_in_len"]

# pre-tokenize data to avoid tokenizing twice
def tokenize(sample):
    sample["tokens"] = gpt2_tokenizer.encode(sample["review"])[:input_len]
    sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
    return sample


ds = ds.map(tokenize, batched=False)

# make dataloader
def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


dataloader = torch.utils.data.DataLoader(
    ds, batch_size=config["batch_size"], collate_fn=collater
)

# %%
# make objective function
def score_response(responses: list[str], target_word: str):
    # Regex checking whether string contains target_word
    regex = re.compile(rf"\b{target_word}\b", re.IGNORECASE)
    scores = torch.zeros(len(responses))
    for i, response in enumerate(responses):
        if regex.search(response):
            scores[i] = 10.0
    return scores


# %%
# initialize trainer
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)

target_word = "movie"
# get datetime
now = datetime.now()
run_name = f"trl-{model_name}-{target_word}-{now.strftime('%Y-%m-%d-%H-%M-%S')}"

all_config = config.copy()
all_config.update(gen_kwargs)

wandb.init(
    project="lmrl",
    config=all_config,
    name=run_name,
)
wandb.watch(gpt2_model, log="all")

total_epochs = config["steps"] // len(ds)
steps_per_epoch = len(ds) // config["batch_size"]

for epoch in range(total_epochs):
    print(f"Epoch {epoch}")
    for step, batch in tqdm(zip(range(steps_per_epoch), iter(dataloader))):
        print(f"Step {step}")
        logs, timing = dict(), dict()
        t0 = time.time()
        query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

        #### Get response from gpt2
        t = time.time()
        response_tensors = []
        batch["response"] = []

        n = config["batch_size"] // config["minibatch_size"]
        gen_len = config["txt_out_len"]
        for i in range(n):
            # fix input and output length so that we can batch things
            queries = torch.stack(query_tensors)[
                i * config["minibatch_size"] : (i + 1) * config["minibatch_size"],
                :input_len,
            ]
            responses = gpt2_model.generate(queries, max_length=gen_len, **gen_kwargs)[
                :, input_len:
            ]
            response_tensors.extend([r.squeeze() for r in responses])
            batch["response"].extend(
                [gpt2_tokenizer.decode(r.squeeze()) for r in responses]
            )
            timing["time/get_response"] = time.time() - t

        print("done generating")
        #### Compute score
        t = time.time()
        rewards = score_response(batch["response"], target_word).to(device)
        timing["time/get_rewards"] = time.time() - t

        #### Run PPO step
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        timing["time/optimization"] = time.time() - t

        #### Log everything
        timing["time/epoch"] = time.time() - t0
        table_rows = [
            list(r)
            for r in zip(batch["query"], batch["response"], rewards.cpu().tolist())
        ]
        logs.update(
            {
                "game_log": wandb.Table(
                    columns=["query", "response", "reward"], rows=table_rows
                )
            }
        )
        logs.update(timing)
        logs.update(stats)
        logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy()
        logs["env/reward_std"] = torch.std(rewards).cpu().numpy()
        logs["env/reward_dist"] = rewards.cpu().numpy()
        try:
            wandb.log(logs)
        except:
            print(logs)
            print("an error occurred, skipped logging")

        if epoch % 1 == 0:
            print(f"Example query: {gpt2_tokenizer.decode(query_tensors[0])}")
            print(f"Example response: {batch['response'][0]}")
            print(f"Reward: {rewards[0]}")
            print(f"Mean reward: {torch.mean(rewards).cpu().item()}")
            torch.save(gpt2_model.state_dict(), f"gpt2-{run_name}.pt")

# %%
