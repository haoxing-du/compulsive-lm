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

# %%
config = {
    "steps": 20000,
    "batch_size": 256,
    "forward_batch_size": 16,
    "txt_in_min_len": 16,
    "txt_in_max_len": 16,
    "txt_out_min_len": 32,
    "txt_out_max_len": 32,
    "lr": 1e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": 0.2,
    "cliprange_value": 0.2,
    "vf_coef": 0.1,
}
# %%
# get models

gpt2_model = GPT2HeadWithValueModel.from_pretrained("gpt2")
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)
gpt2_model_ref.to(device)

gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
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
class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)


input_size = LengthSampler(config["txt_in_min_len"], config["txt_in_max_len"])
output_size = LengthSampler(config["txt_out_min_len"], config["txt_out_max_len"])

input_len = config["txt_in_max_len"]

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
run_name = f"trl-{target_word}-{now.strftime('%Y-%m-%d-%H-%M-%S')}"

wandb.init(
    project="lmrl",
    config=config,
    name=run_name,
)
wandb.watch(gpt2_model, log="all")

total_ppo_epochs = config["steps"] // config["batch_size"]

for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):
    print(f"Epoch {epoch}")
    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

    #### Get response from gpt2
    t = time.time()
    response_tensors = []

    # fix input and output length so that we can batch things
    gen_len = config["txt_out_max_len"]
    queries = torch.stack(query_tensors)[:, :input_len]
    responses = gpt2_model.generate(queries, max_length=gen_len, **gen_kwargs)[
        :, input_len:
    ]
    response_tensors = [r.squeeze() for r in responses]
    batch["response"] = [gpt2_tokenizer.decode(r.squeeze()) for r in responses]

    # for i in range(config["batch_size"]):
    #     gen_len = output_size()
    #     in_len = len(query_tensors[i])
    #     response = gpt2_model.generate(
    #         query_tensors[i].unsqueeze(dim=0), max_new_tokens=gen_len, **gen_kwargs
    #     )
    #     response_tensors.append(response.squeeze()[in_len:])
    # batch["response"] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing["time/get_response"] = time.time() - t

    #### Compute score
    t = time.time()
    rewards = score_response(batch["response"], target_word).to(device)
    # print(rewards)
    # rewards = torch.tensor([1.0 for _ in range(len(query_tensors))])
    # texts = [q + r for q,r in zip(batch['query'], batch['response'])]
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device)
    timing["time/get_rewards"] = time.time() - t

    #### Run PPO step
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    # print(gpt2_model.base_model.h[0].mlp.c_fc.weight[0,:10])
    timing["time/optimization"] = time.time() - t

    #### Log everything
    timing["time/epoch"] = time.time() - t0
    table_rows = [
        list(r) for r in zip(batch["query"], batch["response"], rewards.cpu().tolist())
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
        print(f"Query: {gpt2_tokenizer.decode(query_tensors[0])}")
        print(f"Response: {batch['response'][0]}")
        print(f"Reward: {rewards[0]}")
        torch.save(gpt2_model.state_dict(), f"gpt2-{run_name}.pt")

    # assert torch.allclose(
    #     gpt2_model.base_model.h[0].mlp.c_fc.weight,
    #     gpt2_model_ref.base_model.h[0].mlp.c_fc.weight,
    # )
    # assert torch.allclose(
    #     gpt2_model.base_model.ln_f.weight, gpt2_model_ref.base_model.ln_f.weight
    # )

# %%
