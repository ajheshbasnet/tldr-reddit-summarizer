import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

"""# **REINFORCEMENT LEARNING**"""
datasets = load_dataset("CarperAI/openai_summarize_comparisons")
datasets

run = wandb.init(
    project="reinforcement-learning-fine-tuning",
    name="direct policy optimization",
    config=vars(config)
)

import copy

FrozenModel = copy.deepcopy(model)

for p in FrozenModel.tokenEmbddding.parameters():
    p.requires_grad = False

for stack in FrozenModel.decoderStack:
    for params in stack.parameters():
        params.requires_grad = False

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TOTAL_TRAINING_STEPS = 20000


def lr_lambda(steps):
    MIN_LEARNING_RATE = 5e-6
    WARMUP_STEP = 250
    TOTAL_STEP = TOTAL_TRAINING_STEPS

    if steps < WARMUP_STEP:
        return steps/WARMUP_STEP

    progress = (steps - WARMUP_STEP) / (TOTAL_STEP- WARMUP_STEP)

    return MIN_LEARNING_RATE + 0.5 * (1 + math.cos(math.pi * progress))


LEARNING_RATE: float = 5e-4
optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ======================== CONFIG ======================== #
TRAIN_DATA = datasets["train"]
VALID_DATA = datasets["valid1"]

GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0
PRINT_STEP = 500
SAVING_STEP = 1000
VALIDATION_STEP = 500

# ======================== TRAINING ======================== #
optimizer.zero_grad()

for step in range(TOTAL_TRAINING_STEPS):
    # --- Sample batch indices ---
    indices = torch.randperm(TRAIN_DATA.num_rows)
    start_index = step * TRAIN_BATCH_SIZE
    end_index = start_index + TRAIN_BATCH_SIZE
    batch_indices = indices[start_index:end_index]

    batch_loss = 0.0

    # --- Loop through batch ---
    for idx in batch_indices:
        idx = idx.item()

        prompt_ids = tokenizer(TRAIN_DATA["prompt"][idx])["input_ids"]
        chosen_ids = tokenizer(TRAIN_DATA["chosen"][idx])["input_ids"]
        rejected_ids = tokenizer(TRAIN_DATA["rejected"][idx])["input_ids"]

        # Compute loss
        loss = model.calculateLoss(prompt_ids, chosen_ids, rejected_ids, FrozenModel)
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        batch_loss += loss.item()

    # --- Gradient accumulation step ---
    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # --- Logging ---
    wandb.log({
        "train-loss-per-batch": batch_loss,
        "current-learning-rate": optimizer.param_groups[0]['lr']
    })

    # --- Validation ---
    if (step + 1) % VALIDATION_STEP == 0:
        val_indices = torch.randperm(VALID_DATA.num_rows)
        total_val_loss = 0.0
        num_val_batches = 0
        VALID_STEP = 250  # number of validation mini-batches

        for v in range(VALID_STEP):
            start = v * VALID_BATCH_SIZE
            end = start + VALID_BATCH_SIZE
            batch_val_indices = val_indices[start:end]

            batch_val_loss = 0.0
            for idx in batch_val_indices:
                idx = idx.item()
                prompt_ids = tokenizer(VALID_DATA["prompt"][idx])["input_ids"]
                chosen_ids = tokenizer(VALID_DATA["chosen"][idx])["input_ids"] + [tokenizer.eos_token_id]
                rejected_ids = tokenizer(VALID_DATA["rejected"][idx])["input_ids"] + [tokenizer.eos_token_id]

                val_loss = model.calculateLoss(prompt_ids, chosen_ids, rejected_ids, FrozenModel)
                batch_val_loss += val_loss.item()

            batch_val_loss /= VALID_BATCH_SIZE
            total_val_loss += batch_val_loss
            num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        wandb.log({"valid-loss-per-batch": avg_val_loss})

        if step % PRINT_STEP == 0:
            print(f"[Step {step}] Validation Loss = {avg_val_loss:.4f}")

        if step % SAVING_STEP == 0:
            torch.save(model.state_dict(), f"weights-{step}.pt")

rejectCompletion = tokenizer(datasets['train']["rejected"][15])['input_ids']
print(rejectCompletion)
print(len(rejectCompletion))

torch.save(model.state_dict(), "RL-weights.pt")
