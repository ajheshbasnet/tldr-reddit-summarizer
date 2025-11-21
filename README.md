# TL;DR Reddit Summarizer

This repository contains a compact, efficient decoder-only Transformer model trained to generate concise **TL;DR summaries** for Reddit posts. The model is built with modern architectural components such as **KV-cache**, **RoPE**, and **Grouped Query Attention (GQA)**, enabling fast inference and stable training on modest hardware.

The training pipeline follows three stages:

1. **Pre-training** – General next-token prediction to learn core language patterns.  
2. **Supervised Fine-Tuning (SFT)** – Training on Reddit post → summary pairs to learn summarization behavior.  
3. **Direct Preference Optimization (DPO)** – Improving summary quality using preferred vs. rejected examples.

The result is a lightweight model capable of producing short, clear, human-style TL;DR summaries.

---

## Model Architecture

- Decoder-only Transformer  
- Rotary Positional Embeddings (RoPE)  
- Grouped Query Attention (GQA)  
- KV-cache for fast autoregressive generation  
- Pre-norm Transformer blocks  
- 12 layers, optimized for Reddit summarization  

This setup balances performance and efficiency, making it suitable for single-GPU training and deployment.

---

## Training Environment

Training was performed on a **single NVIDIA T4 GPU**.  
Gradient accumulation is used to increase the effective batch size without exceeding memory limits.

---

## Configuration

All major model and training hyperparameters used in this project:

```text
max_seq_len: 750
d_model: 512
n_heads: 8
d_head: d_model // n_heads
n_groups: 4
vocabs_size: 50257

layers: 12
epochs: 11

train_batch_size: 12
valid_batch_size: 4
gradient_accumulation_size: 4
