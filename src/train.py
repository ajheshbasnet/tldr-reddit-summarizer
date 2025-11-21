import math
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
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
    epochs: int = 11
    layers: int = 12
    train_batch_size: int = 12
    valid_batch_size: int = 4
    gradient_accumulation_size: int = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = ModelConfig()


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


## =============== TOKEN EMBEDDINGS =============== ##

class TokenEmbedding(nn.Module):

    def  __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocabs_size, config.d_model)

    def forward(self, x):
        x = self.embeddings(x)
        return x

## =============== ROTARY POSITION ENCODING =============== ##

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
          angle = self.angle[pos, ...].unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.config.device)               # [1, 1, 1, d_head//2]
        else:
          angle = self.angle[:T, ...].unsqueeze(0).unsqueeze(0).to(self.config.device)                             # [1, 1, T, d_head//2]
        x1, y1 = x[..., ::2], x[..., 1::2]
        x_rot_even, x_rot_odd = (x1*torch.cos(angle) - y1*torch.sin(angle)), (x1*torch.sin(angle) + y1*torch.cos(angle))
        x_out = torch.empty_like(x)
        x_out[..., ::2], x_out[..., 1::2] = x_rot_even, x_rot_odd
        return x_out

## =============== GROUPED QUERY ATTENTION WITH ROPE + KV-CACHE =============== ##

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
                queries = self.rope(queries, self.k_cache.size(2))
                keys = self.rope(keys, self.k_cache.size(2))

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

## =============== RMS NORMALIZATION =============== ##

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

## =============== A DECODER BLOCK =============== ##

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

## =============== LANGUAGE MODEL =============== ##

class MyLanguageModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
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

    def generate(self, ids, max_seq_len, temperature: float= 1.0):

        for layer in self.decoderStack:
          layer.reset_kv_cache()

        ids = ids[:, :self.config.max_seq_len] if len(ids)>self.config.max_seq_len else ids
        ids = torch.tensor(ids, dtype = torch.long).unsqueeze(0).to(self.config.device)
        input_id = ids.clone()
        for i in range(max_seq_len):
            logits, _ = self.forward(input_id, kv_cache = True)
            last_id = logits[:, -1, :]/temperature
            last_id = torch.nn.functional.softmax(last_id, dim = -1)
            token_id = torch.multinomial(last_id, num_samples = 1)
            ids = torch.cat((ids, token_id), dim = 1)
            input_id = token_id

        return ids

model = MyLanguageModel(config).to(config.device)


## =============== FOR INFERENCE =============== ##

def generate_text(text: str, max_length:int=config.max_seq_len):

    ids = tokenizer(text)['input_ids']
    output_ids = model.generate(
        ids,
        max_seq_len=max_length
      )
    text = tokenizer.decode(output_ids[0])
    return text

generate_text("REDDIT", 19)   
'''it's output will be gebrish since the model ain't yet train :(|)'''

## =============== TOTAL NUMBER OF TRAINABLE PARAMETERS =============== ##

print(f'--- << THE TOTAL PARAMETERS OF MY LANGUAGE MODEL IS: {sum(p.numel() for p in model.parameters())/1e6 :.2f} MILLIONS >> ---')

## =============== TRAINING LOOP =============== ##


"""0. Weights Initialization
1. Decaying Learning Rates
2. Gradient Clipping
3. Mixed Precison
4. tdqm
5. Save-CheckPoint
6. Log all the graphs in wandb
"""

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
    WARMUP_STEP:int = 40
    TOTAL_STEP:int = 460

    if steps < WARMUP_STEP:
        return steps/WARMUP_STEP

    progress = (steps - WARMUP_STEP) / (TOTAL_STEP- WARMUP_STEP)
    return MIN_LEARNING_RATE + 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


print(generate_text("TITLE: My boyfriend [M33] and I have been in our relationship but totally honest about herself.", 450))


import matplotlib.pyplot as plt

plt.plot(all_training_loss, marker = 'o')
plt.plot(all_validation_loss, marker = 'o')
plt.grid(True, alpha = 0.4)
plt.plot()