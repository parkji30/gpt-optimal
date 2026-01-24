"""Minimal GPT-2 implementation in PyTorch with training."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    """GPT-2 model configuration."""
    vocab_size: int = 256
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 128
    dropout: float = 0.1

    @classmethod
    def from_json(cls, config_dict: dict) -> "GPT2Config":
        model_cfg = config_dict["model"]
        return cls(
            vocab_size=model_cfg["vocab_size"],
            n_embd=model_cfg["n_embd"],
            n_head=model_cfg["n_head"],
            n_layer=model_cfg["n_layer"],
            block_size=model_cfg["block_size"],
            dropout=model_cfg["dropout"],
        )


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 Language Model."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class CharDataset:
    """Character-level dataset for training."""

    def __init__(self, text: str, block_size: int):
        self.block_size = block_size
        self.data = torch.tensor([ord(c) for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: torch.device):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


@torch.no_grad()
def estimate_loss(model: GPT2, train_data: CharDataset, val_data: CharDataset,
                  eval_iters: int, batch_size: int, device: torch.device) -> dict:
    """Estimate loss on train and val sets."""
    out = {}
    model.eval()
    for name, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data.get_batch(batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean()
    model.train()
    return out


def train(config_path: str):
    """Train GPT-2 model using configuration from JSON file."""
    # Load configuration
    config_dict = load_config(config_path)
    model_config = GPT2Config.from_json(config_dict)
    training_cfg = config_dict["training"]
    data_cfg = config_dict["data"]
    gen_cfg = config_dict["generation"]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    script_dir = Path(__file__).parent
    data_path = script_dir / data_cfg["path"]
    print(f"Loading data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters")

    # Train/val split
    n = int(len(text) * data_cfg["train_split"])
    train_data = CharDataset(text[:n], model_config.block_size)
    val_data = CharDataset(text[n:], model_config.block_size)
    print(f"Train size: {len(train_data):,}, Val size: {len(val_data):,}")

    # Initialize model
    model = GPT2(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg["learning_rate"])

    # Training loop
    print("\nStarting training...")
    training_start = time.time()
    for iter_num in range(training_cfg["max_iters"]):
        # Evaluate periodically
        if iter_num % training_cfg["eval_interval"] == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                training_cfg["eval_iters"],
                training_cfg["batch_size"],
                device
            )
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Get batch and compute loss
        x, y = train_data.get_batch(training_cfg["batch_size"], device)
        _, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    training_duration = time.time() - training_start

    # Final evaluation
    losses = estimate_loss(
        model, train_data, val_data,
        training_cfg["eval_iters"],
        training_cfg["batch_size"],
        device
    )
    print(f"\nFinal: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Generate sample
    print("\nGenerating sample...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"]
    )
    output_text = ''.join([chr(i) for i in generated[0].tolist()])
    print("=" * 50)
    print(output_text)
    print("=" * 50)

    # Save model
    save_path = script_dir / "model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")
    print(f"Total training time: {training_duration:.2f}s")


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "../hyperparams/config.json"
    train(str(config_path))


if __name__ == "__main__":
    main()
