"""Minimal GPT-2 implementation in PyTorch with training and modern optimizations."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader


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


def setup_cuda_optimizations(opt_cfg: dict):
    """Configure CUDA backend optimizations."""
    if not torch.cuda.is_available():
        return

    if opt_cfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    if opt_cfg.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if opt_cfg.get("flash_attention", True):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention support."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Flash Attention with scaled_dot_product_attention
        # Automatically selects optimal backend (Flash, Memory-Efficient, or Math)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)

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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
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
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
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


class CharDataset(Dataset):
    """Character-level dataset for training with proper DataLoader support."""

    def __init__(self, text: str, block_size: int):
        self.block_size = block_size
        self.data = torch.tensor([ord(c) for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def create_dataloader(dataset: CharDataset, batch_size: int, opt_cfg: dict, device: torch.device) -> DataLoader:
    """Create optimized DataLoader with pinned memory and workers."""
    use_cuda = device.type == 'cuda'
    num_workers = opt_cfg.get("num_workers", 2) if use_cuda else 0
    persistent_workers = opt_cfg.get("persistent_workers", True) and num_workers > 0
    prefetch_factor = opt_cfg.get("prefetch_factor", 2) if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=opt_cfg.get("pin_memory", True) and use_cuda,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_lr(iter_num: int, warmup_iters: int, lr_decay_iters: int, learning_rate: float, min_lr: float) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    # Linear warmup
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / warmup_iters
    # Cosine decay after warmup
    if iter_num > lr_decay_iters:
        return min_lr
    # Cosine annealing
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT2, train_loader: DataLoader, val_loader: DataLoader,
                  eval_iters: int, device: torch.device, use_amp: bool) -> dict:
    """Estimate loss on train and val sets."""
    out = {}
    model.eval()
    for name, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        data_iter = iter(loader)
        for k in range(eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=use_amp):
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
    amp_cfg = config_dict.get("amp", {})
    opt_cfg = config_dict.get("optimization", {})

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup CUDA optimizations (Phase 1)
    setup_cuda_optimizations(opt_cfg)

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

    # Create DataLoaders (Phase 4)
    train_loader = create_dataloader(train_data, training_cfg["batch_size"], opt_cfg, device)
    val_loader = create_dataloader(val_data, training_cfg["batch_size"], opt_cfg, device)

    # Initialize model
    model = GPT2(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Compile model (Phase 5)
    if opt_cfg.get("compile", True) and device.type == 'cuda':
        print("Compiling model with torch.compile()...")
        compile_mode = opt_cfg.get("compile_mode", "default")
        model = torch.compile(model, mode=compile_mode, dynamic=False)

    # Optimizer with fused AdamW (Phase 6)
    use_fused = opt_cfg.get("fused_optimizer", True) and device.type == 'cuda'
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        fused=use_fused
    )

    # AMP setup (Phase 3)
    use_amp = amp_cfg.get("enabled", True) and device.type == 'cuda'
    scaler = GradScaler(
        device='cuda',
        init_scale=amp_cfg.get("init_scale", 65536),
        growth_factor=amp_cfg.get("growth_factor", 2.0),
        backoff_factor=amp_cfg.get("backoff_factor", 0.5),
        growth_interval=amp_cfg.get("growth_interval", 2000),
        enabled=use_amp
    )

    # Training parameters
    gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
    warmup_iters = training_cfg.get("warmup_iters", 100)
    lr_decay_iters = training_cfg.get("lr_decay_iters", training_cfg["max_iters"])
    min_lr = training_cfg.get("min_lr", training_cfg["learning_rate"] / 10)

    # Training loop
    print("\nStarting training...")
    train_iter = iter(train_loader)

    # Warmup for torch.compile() - first iterations trigger compilation
    print("Warming up (triggering compilation)...")
    warmup_start = time.time()
    for warmup_iter in range(3):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device_type='cuda', enabled=use_amp):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start
    print(f"Warmup complete ({warmup_time:.2f}s including compilation)")

    # Reset iterator and start actual training
    train_iter = iter(train_loader)
    training_start = time.time()

    for iter_num in range(training_cfg["max_iters"]):
        # Update learning rate (Phase 6)
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, training_cfg["learning_rate"], min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate periodically
        if iter_num % training_cfg["eval_interval"] == 0:
            losses = estimate_loss(
                model, train_loader, val_loader,
                training_cfg["eval_iters"],
                device, use_amp
            )
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Gradient accumulation loop (Phase 7)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(gradient_accumulation_steps):
            # Get batch from DataLoader
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward pass with AMP (Phase 3)
            with autocast(device_type='cuda', enabled=use_amp):
                _, loss = model(x, y)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            accumulated_loss += loss.item()

            # Backward pass with scaler
            scaler.scale(loss).backward()

        # Gradient clipping (Phase 7)
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

    training_duration = time.time() - training_start

    # Final evaluation
    losses = estimate_loss(
        model, train_loader, val_loader,
        training_cfg["eval_iters"],
        device, use_amp
    )
    print(f"\nFinal: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Generate sample
    print("\nGenerating sample...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
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
    # Handle compiled model state dict
    if hasattr(model, '_orig_mod'):
        torch.save(model._orig_mod.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")
    print(f"Total training time: {training_duration:.2f}s")

    # Memory stats
    if device.type == 'cuda':
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {max_mem:.1f} MB")


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "../hyperparams/config.json"
    train(str(config_path))


if __name__ == "__main__":
    main()
