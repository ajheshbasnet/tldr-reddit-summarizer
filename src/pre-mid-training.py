import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

@dataclass
class ModelConfig():
    max_seq_len: int = 750
    d_model: int = 512
    n_heads: int = 8
    assert d_model % n_heads==0, "d_model should be divisible by n_heads"
    d_head: int = d_model // n_heads
    n_groups: int = 4
    vocabs_size: int = 50257
    epochs: int = 13
    layers: int = 10
    train_batch_size: int = 8
    valid_batch_size: int = 4
    gradient_accumulation_size: int = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = ModelConfig()

"""# **DATASET**"""

dataset = load_dataset("CarperAI/openai_summarize_tldr")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id


class MyCustomDataset(Dataset):

    def __init__(self, data, purpose: str):
      self.train = data[purpose]


    def __len__(self):
      return len(self.train)

    def __getitem__(self, idx):
      txt = self.train['prompt'][idx]
      self.ids = tokenizer(txt, max_length=config.max_seq_len+1, padding = 'max_length', truncation = True)['input_ids']
      x = self.ids[:-1]
      y = self.ids[1:]

      return {
          'input': torch.tensor(x, dtype=torch.long),
          'target': torch.tensor(y, dtype=torch.long)
              }

trainMyDataset = MyCustomDataset(dataset, "train")
validMyDataset = MyCustomDataset(dataset, "valid")

train_data = DataLoader(trainMyDataset, batch_size=config.train_batch_size, shuffle=True)
valid_data = DataLoader(validMyDataset, batch_size=config.valid_batch_size, shuffle=False)

"""# **MODEL**

1. RoPE
2. MultiQueryAttention
3. KV-Cache
4. Dropouts
5. Layer-Normalization
"""

class TokenEmbedding(nn.Module):

    def  __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocabs_size, config.d_model)

    def forward(self, x):
        x = self.embeddings(x)
        return x

"""# **POSITIONAL-ENCODING**"""


class RotaryPositionEncoding(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.config = config
        assert self.config.d_head % 2 == 0, "the dimension should be divisible by 2"

        pos = torch.arange(0, self.config.max_seq_len, device=self.config.device).unsqueeze(-1)
        frequency = torch.exp(-(torch.arange(0, self.config.d_head,2, device=self.config.device)/self.config.d_head) * torch.log(torch.tensor(10000., device = self.config.device)))
        self.angle = pos * frequency           # [T, D_head/2]

    def forward(self, x, pos = None):

        B, n_head, T, C = x.size()                                                                                 # [B, n_head, T, d_head]
        if pos is not None:
          #angle = self.angle[pos, ...].unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.config.device)               # [1, 1, 1, d_head//2]
          angle = self.angle[pos].unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.config.device)               # [1, 1, 1, d_head//2]
        else:
          angle = self.angle[:T, ...].unsqueeze(0).unsqueeze(0).to(self.config.device)                             # [1, 1, T, d_head//2]
        x1, y1 = x[..., ::2], x[..., 1::2]
        x_rot_even, x_rot_odd = (x1*torch.cos(angle) - y1*torch.sin(angle)), (x1*torch.sin(angle) + y1*torch.cos(angle))
        x_out = torch.empty_like(x)
        x_out[..., ::2], x_out[..., 1::2] = x_rot_even, x_rot_odd
        return x_out

class GQA(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.heads_per_group = self.config.n_heads//self.config.n_groups
        self.d_group = self.config.n_groups * self.config.d_head

        self.query = nn.Linear(self.config.d_model, self.config.d_model)
        self.key = nn.Linear(self.config.d_model, self.d_group)
        self.value = nn.Linear(self.config.d_model, self.d_group)

        self.output = nn.Linear(self.config.d_model, self.config.d_model)
        self.rope = RotaryPositionEncoding(self.config)
        self.dropout = nn.Dropout(0.05)

        self.k_cache = None
        self.v_cache = None

    def forward(self, x, kv_cache = False):

        que = self.query(x)
        key = self.key(x)
        val = self.value(x)

        scale = math.sqrt(self.config.d_head)

        B, T, C = x.size()   # [1, 1, 256]

        queries = que.view(B, T, self.config.n_heads, self.config.d_head).permute(0, 2, 1, 3)
        keys = key.view(B, T, self.config.n_groups, self.config.d_head).permute(0, 2, 1, 3)
        values = val.view(B, T, self.config.n_groups, self.config.d_head).permute(0, 2, 1, 3)

        idx = torch.arange(self.config.n_heads, device=self.config.device)//self.heads_per_group

        keys = keys[:, idx, :, :]
        values = values[:, idx, :, :]

        if kv_cache:
          with torch.no_grad():
            B, T, C = x.size()   #[1, 1, 256]
            if self.k_cache is None:
                queries = self.rope(queries)
                keys = self.rope(keys)

                self.k_cache = keys.detach().clone()
                self.v_cache = values.detach().clone()
            else:
                queries = self.rope(queries, self.k_cache.size(2)+1)
                keys = self.rope(keys, self.k_cache.size(2)+1)

                self.k_cache = torch.cat((self.k_cache, keys.detach().clone()), dim = 2)
                # .clone() so that the computational graph of kv-cache is not computed.
                self.v_cache = torch.cat((self.v_cache, values.detach().clone()), dim = 2)

            attn_score = (queries @ self.k_cache.transpose(-2, -1))/scale
            attn_score = torch.nn.functional.softmax(attn_score, dim = -1)
            attn_score = self.dropout(attn_score)
            attn = attn_score @ self.v_cache

        if not kv_cache:

            queries = self.rope(queries)
            keys = self.rope(keys)

            B, T, C = x.size()

            attn_score = (queries @ keys.transpose(-2, -1)) / scale

            masked = torch.triu(torch.ones(T, T, device=self.config.device), diagonal = 1)
            masked = masked.masked_fill(masked==1, float('-inf'))

            attn_probs = attn_score + masked
            attn_score = torch.nn.functional.softmax(attn_probs, dim = -1)    # [B, n_H, T, D]
            attn = self.dropout(attn_score)
            attn = attn @ values

        attn = attn.transpose(2,1).reshape(B, T, -1)
        attn = self.output(attn)

        return attn


class RMSNorm(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.config = config
        self.eps = 1e-9
        self.weight = nn.Parameter(torch.ones(self.config.d_model, device=self.config.device))

    def forward(self, x):
        scale = torch.sqrt(torch.mean(x**2, dim = -1, keepdim=True)) + self.eps
        x = x / scale
        return self.weight * x

class SwigLuFFNN(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.config = config
        self.layer1 = nn.Linear(self.config.d_model, 2*self.config.d_model)
        self.layer2 = nn.Linear(config.d_model, self.config.d_model)

    def forward(self, x):
        h1 = self.layer1(x)
        x1, x2 = h1.chunk(2, dim = -1)  ## same as h1[..., :d_model], h1[..., d_model:]
        act = x2 * torch.sigmoid(x2)
        x = x1 * act
        x = self.layer2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config):

      super().__init__()
      self.config = config
      self.MultiAttention = GQA(self.config)
      self.SwigLuFFNN = SwigLuFFNN(self.config)
      self.RMSNorm1 = RMSNorm(self.config)
      self.RMSNorm2 = RMSNorm(self.config)


    def reset_kv_cache(self):
      self.MultiAttention.k_cache = None
      self.MultiAttention.v_cache = None

    def forward(self, x, kv_cache = False):
      res = x
      x = self.RMSNorm1(x)
      x = self.MultiAttention(x, kv_cache) + res

      res = x
      x = self.RMSNorm2(x)
      x = self.SwigLuFFNN(x) + res
      return x

class MyLanguageModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.max_seq_len = config.max_seq_len
        self.tokenEmbddding = TokenEmbedding(self.config)
        self.decoderStack = nn.ModuleList([Decoder(self.config) for _ in range(self.config.layers)])
        self.projectionLayer = nn.Linear(self.config.d_model, self.config.vocabs_size)
        self.rmsNorm = RMSNorm(self.config)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, target = None, kv_cache=False):
        x = self.tokenEmbddding(x)
        for layer in self.decoderStack:
          x = layer(x, kv_cache)
        x = self.rmsNorm(x)
        logits = self.projectionLayer(x)
        if target is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.config.vocabs_size),
                target.view(-1),
                ignore_index = tokenizer.eos_token_id
            )
        else:
            loss = None
        return logits, loss

    def generate(self, ids, max_seq_len = 750, temperature: float= 1.0):

        for layer in self.decoderStack:
          layer.reset_kv_cache()
        ids = ids[:, :self.config.max_seq_len] if len(ids)>=self.config.max_seq_len else ids    # 500
        ids = torch.tensor(ids, dtype = torch.long).unsqueeze(0).to(self.config.device)
        input_id = ids.clone()

        if max_seq_len == self.config.max_seq_len:
            max_token_gen_len = self.config.max_seq_len - len(ids[0])

        elif (len(ids[0]) + max_seq_len < self.config.max_seq_len):
            max_token_gen_len = max_seq_len

        for i in range(max_token_gen_len):
            logits, _ = self.forward(input_id, kv_cache = True)
            last_id = logits[:, -1, :]/temperature
            last_id = torch.nn.functional.softmax(last_id, dim = -1)
            token_id = torch.multinomial(last_id, num_samples = 1)
            ids = torch.cat((ids, token_id), dim = 1)
            input_id = token_id
        return ids

    def computeLog(self, logits, tokens):
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return token_logprobs.sum()   # important

    def calculateLoss(self, x, choosen, rejected, FrozenModel):

        beta = 0.05
        promptLen = len(x)

        inputWchosen = torch.tensor(x+choosen, dtype=torch.long).unsqueeze(0).to(self.config.device)
        inputWreject = torch.tensor(x+rejected, dtype=torch.long).unsqueeze(0).to(self.config.device)

        # 1. Run training model on chosen and rejected
        logits_chosen = self.forward(inputWchosen)[0][:, promptLen:, :]
        logits_rejected = self.forward(inputWreject)[0][:, promptLen:, :]

        train_chosen = self.computeLog(logits_chosen, choosen)
        train_rejected = self.computeLog(logits_rejected, rejected)

        # 2. Run reference model on chosen and rejected
        logits_chosen_ref = FrozenModel.forward(inputWchosen)[0][:, promptLen:, :]
        logits_rejected_ref = FrozenModel.forward(inputWreject)[0][:, promptLen:, :]

        ref_chosen = self.computeLog(logits_chosen_ref, choosen)
        ref_rejected = self.computeLog(logits_rejected_ref, rejected)

        # 3. DPO Δ
        delta = (train_chosen - ref_chosen) - (train_rejected - ref_rejected)

        # 4. DPO loss
        loss = -torch.nn.functional.logsigmoid(beta * delta)
        return loss

model = MyLanguageModel(config).to(config.device)

def generate_text(text: str, max_length, inference = False):

    ids= tokenizer(text)['input_ids']
    output_ids = model.generate(
        ids,
        max_seq_len=max_length
      )
    output_ids = output_ids[0]

    if inference:
        output_id = []
        for ids in output_ids:
            output_id.append(ids)
            if ids == tokenizer.eos_token_id:
                text = tokenizer.decode(output_id)
                return text
                break

        text = tokenizer.decode(output_id)
        return text
    else:
        text = tokenizer.decode(output_ids)
    return text

print(generate_text(dataset['train']['prompt'][-1], 750))

"""# **TRANING LOOP**
0. Weights Initialization
1. Decaying Learning Rates
2. Gradient Clipping
3. Mixed Precison
4. tdqm
5. Save-CheckPoint
6. Log all the graphs in wandb
"""

import wandb
wandb.login(key="XXX")

LEARNING_RATE: float = 2e-3
optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

config_dict = vars(config)

run = wandb.init(
    project="PRE-TRAINING",                    # your project name
    name="training-visualization",             # optional run name
    config=config_dict,                        # pass your config dictionary
)

scaler = torch.amp.GradScaler(config.device)

def lr_lambda(steps):
    MIN_LEARNING_RATE: float = 1e-5
    WARMUP_STEP:int = 850
    TOTAL_STEP:int = int((len(train_data) * config.train_batch_size) / (config.train_batch_size * config.gradient_accumulation_size)) * config.epochs

    if steps < WARMUP_STEP:
        return steps/WARMUP_STEP

    progress = (steps - WARMUP_STEP) / (TOTAL_STEP- WARMUP_STEP)
    return MIN_LEARNING_RATE + 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

from tqdm import tqdm

all_training_loss = []
all_validation_loss = []

for epoch in range(config.epochs):
    model.train()
    running_train_loss = 0.0
    train_samples = 0

    train_progress = tqdm(train_data, desc=f"Epoch {epoch+1}/{config.epochs}")

    # Critical: Reset gradients at epoch start
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(train_progress):
        X, y = batch['input'], batch['target']
        X, y = X.to(config.device), y.to(config.device)

        # Forward pass with autocast
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, loss = model(X, y)

        # Scale loss for gradient accumulation
        # This normalizes gradients as if we had a larger batch
        loss_scaled = loss / config.gradient_accumulation_size
        scaler.scale(loss_scaled).backward()

        # Track loss for reporting (use unscaled loss)
        batch_size = X.size(0)
        running_train_loss += loss.item() * batch_size
        train_samples += batch_size

        # Perform optimizer step after accumulating enough gradients
        if (i + 1) % config.gradient_accumulation_size == 0:
            # Unscale gradients (convert from scaled FP16 to FP32)
            scaler.unscale_(optimizer)

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with scaled gradients
            scaler.step(optimizer)

            # Update scaler for next iteration
            scaler.update()

            # Learning rate schedule step
            scheduler.step()

            # Zero gradients for next accumulation window
            optimizer.zero_grad(set_to_none=True)

        # Update progress bar with current and average loss
        avg_train_loss = running_train_loss / train_samples
        train_progress.set_postfix(
            loss=f"{loss.item():.4f}",
            avg_loss=f"{avg_train_loss:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

        run.log({"learning_rate": optimizer.param_groups[0]['lr'], "training_loss_per_batch": avg_train_loss})

    # Handle remaining accumulated gradients at epoch end
    # This ensures orphaned micro-batches get processed
    if (i + 1) % config.gradient_accumulation_size != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # Final epoch training loss
    avg_train_loss = running_train_loss / train_samples

    # ---------- Validation ----------
    model.eval()
    running_val_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        val_progress = tqdm(valid_data, desc="Validation")
        for batch in val_progress:
            X, y = batch['input'], batch['target']
            X, y = X.to(config.device), y.to(config.device)

            # No autocast needed in eval, but doesn't hurt
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits, loss = model(X, y)

            batch_size = X.size(0)
            running_val_loss += loss.item() * batch_size
            val_samples += batch_size
            avg_val_loss = running_val_loss / val_samples

            val_progress.set_postfix(val_loss=f"{avg_val_loss:.4f}")

    avg_val_loss = running_val_loss / val_samples

    # Logging
    print(f"\nEpoch [{epoch+1}/{config.epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    all_training_loss.append(avg_train_loss)
    all_validation_loss.append(avg_val_loss)

    run.log({"train-loss-per-epoch":avg_train_loss, "val-loss-per-epoch": avg_val_loss})

    file_name = f'/kaggle/working/epoch: {epoch+1}.pt'

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, file_name)
    print(f"✓ Best model saved (val_loss: {avg_val_loss:.4f})")
run.finish()

generate_text("TITLE: My boyfriend [M33] and I have been in our relationship but totally honest about herself.", 450, False)

import matplotlib.pyplot as plt

plt.plot(all_training_loss, marker = 'o')
plt.plot(all_validation_loss, marker = 'o')
plt.grid(True, alpha = 0.4)
plt.plot()

##================= MID TRAINING =================##



model.load_state_dict(torch.load("/kaggle/input/end-sft-weights/pytorch/default/1/model-weights-5.pt", map_location = config.device))

print(generate_text(dataset['train']['prompt'][-2], 750, True))

print(dataset['train']['label'][-2])

class SFT(Dataset):

    def __init__(self, config, datas, *, SUPERVISE_DATA_LEN:int, purpose: str):

        self.config = config
        perm = np.random.permutation(dataset[purpose].num_rows)[:SUPERVISE_DATA_LEN]
        self.prompts = datas[purpose]['prompt'][perm]
        self.labels = datas[purpose]['label'][perm]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):

        if idx<=SUPERVISE_DATA_LEN:

            inputs = self.prompts[idx]
            target = self.labels[idx]

            input_ids = tokenizer(inputs)['input_ids']
            input_mask = [True] * len(input_ids)

            target_ids = tokenizer(target)['input_ids']
            target_mask = [False] * len(target_ids)

            ids = input_ids + target_ids
            mask = input_mask + target_mask

            if len(ids) <= (self.config.max_seq_len+1):
                ids = ids + [tokenizer.eos_token_id]*(self.config.max_seq_len - len(ids) + 1)
                mask = mask + [False]+([True]*(self.config.max_seq_len - len(mask)))

            if len(ids) > (self.config.max_seq_len+1):
                ids = ids + [tokenizer.eos_token_id]
                mask = mask + [False]
                ids = ids[self.config.max_seq_len - 1:]
                mask = mask[self.config.max_seq_len:]

            input_ids = torch.tensor(ids[:-1], dtype = torch.long)
            target_ids = torch.tensor(ids[1:], dtype = torch.long)

            target_mask = mask[1:]

            target_ids[target_mask] = -100

            return {
                "input_ids": input_ids,
                "target_ids": target_ids,
            }

SUPERVISE_DATA_LEN=45000

train_dataset = SFT(config, dataset, SUPERVISE_DATA_LEN=45000, purpose="train")
val_dataset = SFT(config, dataset, SUPERVISE_DATA_LEN=1500, purpose="valid")

train_data = DataLoader(train_dataset, batch_size = config.train_batch_size, shuffle = True)
valid_data = DataLoader(val_dataset, batch_size = config.valid_batch_size, shuffle = False)
print(config.train_batch_size)

run = wandb.init(
    project="Training-Language-Model",
    name="SuperVised-Fine-Tuning",
    config=config_dict
)

SFT_EPOCHS = 3
def lr_lambda(steps):
    MIN_LEARNING_RATE = 5e-7
    WARMUP_STEP = 250
    TOTAL_STEP = math.ceil(SUPERVISE_DATA_LEN/(config.train_batch_size * config.gradient_accumulation_size))*SFT_EPOCHS

    if steps < WARMUP_STEP:
        return steps/WARMUP_STEP

    progress = (steps - WARMUP_STEP) / (TOTAL_STEP- WARMUP_STEP)
    return MIN_LEARNING_RATE + 0.5 * (1 + math.cos(math.pi * progress))


LEARNING_RATE: float = 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

all_training_loss = []
all_validation_loss = []

validation_step = 8

for epoch in range(SFT_EPOCHS):
    model.train()
    running_train_loss = 0.0
    train_samples = 0

    train_progress = tqdm(train_data, desc=f"Epoch {epoch+1}/{SFT_EPOCHS}")

    for stack in model.decoderStack[0].parameters():
        stack.requires_grad = False
    for stack in model.decoderStack[1].parameters():
        stack.requires_grad = False

    # Critical: Reset gradients at epoch start
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(train_progress):

        check_i = torch.randperm(len(train_data))[:500]
        if i in check_i:
            continue   # inorder to reduce the over-fitting :)

        X, y = batch['input_ids'], batch['target_ids']
        X, y = X.to(config.device), y.to(config.device)


        # Forward pass with autocast
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, loss = model(X, y)

        # Scale loss for gradient accumulation
        # This normalizes gradients as if we had a larger batch
        loss_scaled = loss / config.gradient_accumulation_size
        scaler.scale(loss_scaled).backward()

        # Track loss for reporting (use unscaled loss)
        batch_size = X.size(0)
        running_train_loss += loss.item() * batch_size
        train_samples += batch_size

        # Perform optimizer step after accumulating enough gradients
        if (i + 1) % config.gradient_accumulation_size == 0:
            # Unscale gradients (convert from scaled FP16 to FP32)
            scaler.unscale_(optimizer)

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with scaled gradients
            scaler.step(optimizer)

            # Update scaler for next iteration
            scaler.update()

            # Learning rate schedule step
            scheduler.step()

            # Zero gradients for next accumulation window
            optimizer.zero_grad(set_to_none=True)

        # Update progress bar with current and average loss
        avg_train_loss = running_train_loss / train_samples
        train_progress.set_postfix(
            loss=f"{loss.item():.4f}",
            avg_loss=f"{avg_train_loss:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

        run.log({"learning_rate": optimizer.param_groups[0]['lr'], "training_loss_per_batch": avg_train_loss})

        if i % validation_step==0:
            # ---------- Validation ----------
            model.eval()
            running_val_loss = 0.0
            val_samples = 0
            total_eos_token = 0

            with torch.no_grad():
                val_progress = tqdm(valid_data, desc="Validation")
                for batch in val_progress:
                    X, y = batch['input'], batch['target']
                    X, y = X.to(config.device), y.to(config.device)

                    # No autocast needed in eval, but doesn't hurt
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits, loss = model(X, y)

                    batch_size = X.size(0)
                    running_val_loss += loss.item() * batch_size
                    val_samples += batch_size
                    avg_val_loss = running_val_loss / val_samples



                    val_progress.set_postfix(val_loss=f"{avg_val_loss:.4f}")


                run.log({"number-of-eos-token-count":avg_train_loss, "val-loss-per-epoch": avg_val_loss})

            avg_val_loss = running_val_loss / val_samples
            model.train()


    # Handle remaining accumulated gradients at epoch end
    # This ensures orphaned micro-batches get processed
    if (i + 1) % config.gradient_accumulation_size != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # Final epoch training loss
    avg_train_loss = running_train_loss / train_samples


    # Logging
    print(f"\nEpoch [{epoch+1}/{config.epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    all_training_loss.append(avg_train_loss)
    all_validation_loss.append(avg_val_loss)

    run.log({"train-loss-per-epoch":avg_train_loss, "val-loss-per-epoch": avg_val_loss})

    # Optional: Model checkpointing
    if avg_val_loss == min(all_validation_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, 'best_model.pt')
        print(f"✓ Best model saved (val_loss: {avg_val_loss:.4f})")
run.finish()
