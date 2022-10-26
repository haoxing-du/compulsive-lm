# %%
# imports
import torch
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
import wandb
from tqdm import tqdm
import time
from datasets import load_dataset
import numpy as np

# %%
config = {
    'steps': 50,
    'batch_size': 1, 
    'forward_batch_size': 1,
    'txt_in_min_len': 2,
    'txt_in_max_len': 8,
    'txt_out_min_len': 4,
    'txt_out_max_len': 16,
    'lr': 1.5e-5,
    }

wandb.init(project='lmrl', config=config)
# %%
# get models

gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)
gpt2_model_ref.to(device)

gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}

# %%
# load imdb with datasets
ds = load_dataset('imdb', split='train')
ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
ds = ds.filter(lambda x: len(x["review"])>200, batched=False)

# randomize the query and response lengths
class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))
    def __call__(self):
        return np.random.choice(self.values)
    
input_size = LengthSampler(config["txt_in_min_len"], config["txt_in_max_len"])
output_size = LengthSampler(config["txt_out_min_len"], config["txt_out_max_len"])

# pre-tokenize data to avoid tokenizing twice
def tokenize(sample):
    sample["tokens"] = gpt2_tokenizer.encode(sample["review"])[:input_size()]
    sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
    return sample

ds = ds.map(tokenize, batched=False)

# make dataloader
def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

dataloader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], collate_fn=collater)

# %%
# initialize trainer
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)

wandb.watch(gpt2_model, log='all')

total_ppo_epochs = config["steps"] // config['batch_size']

for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):
    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
    
    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(config['batch_size']):
        gen_len = output_size()
        response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),
                                       max_new_tokens=gen_len, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time() - t

    #### Compute sentiment score
    t = time.time()
    #score = score_response(response, target_word)
    rewards = torch.ones(query_tensors.shape[0])
    # texts = [q + r for q,r in zip(batch['query'], batch['response'])]
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device)
    timing['time/get_sentiment_preds'] = time.time() - t
    
    #### Run PPO step 
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time() - t
     
    #### Log everything
    timing['time/epoch'] = time.time()-t0
    table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]
    logs.update({'game_log': wandb.Table(columns=['query', 'response', 'reward'], rows=table_rows)})
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    wandb.log(logs)
# %%
# encode a query
query_txt = "This morning I went to the"
query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

# %%
# get model response
response_tensor  = respond_to_batch(gpt2_model, query_tensor)
response_txt = gpt2_tokenizer.decode(response_tensor[0,:])

# %%
# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# %%
# train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
# %%
