# Performance Upgrades Changelog

This file tracks performance optimizations made to the GPT-2 training implementations. Update this file with each commit that improves training speed or efficiency.

---

## 2026-01-24 | `5ebc38b`

### PyTorch Training Loop Optimizations

**Files Modified:**
- `gpt2-python/gpt2.py`
- `hyperparams/config.json`

**Optimizations Implemented:**

| Phase | Feature | Description |
|-------|---------|-------------|
| 1 | CUDA Backend | `cudnn.benchmark=True`, TF32 for matmul, Flash SDP backends enabled |
| 2 | Flash Attention | Replaced manual attention with `F.scaled_dot_product_attention(is_causal=True)` |
| 3 | AMP | `torch.amp.autocast` + `GradScaler` for FP16 mixed precision |
| 4 | DataLoader | `pin_memory=True`, `num_workers=2`, `persistent_workers=True`, `prefetch_factor=2` |
| 5 | torch.compile() | Model compilation with warmup phase to exclude compilation time from benchmarks |
| 6 | Fused AdamW | `torch.optim.AdamW(fused=True)` + LR scheduler with linear warmup and cosine decay |
| 7 | Gradient Accumulation | Support for `gradient_accumulation_steps` + `clip_grad_norm_` |

**Results:**
- Training time: 18.57s (baseline ~21s) - ~12% faster
- Peak GPU memory: 121.3 MB (reduced via Flash Attention)
- Compilation overhead: 13.11s (one-time warmup cost)
- Loss quality maintained: train 2.24, val 2.25

**Config Additions:**
```json
"training": {
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "warmup_iters": 100,
    "lr_decay_iters": 1000,
    "min_lr": 3e-5
},
"optimization": {
    "compile": true,
    "compile_mode": "default",
    "cudnn_benchmark": true,
    "tf32": true,
    "flash_attention": true,
    "fused_optimizer": true,
    "pin_memory": true,
    "num_workers": 2,
    "persistent_workers": true,
    "prefetch_factor": 2
}
```

**Notes:**
- Speedup is modest on small models (842K params). These optimizations provide 2-4x gains on larger models.
- Flash Attention benefits scale with sequence length (O(n) vs O(n^2) memory).
- torch.compile() requires PyTorch 2.0+ and adds one-time compilation overhead.

---

## Template for Future Entries

```markdown
## YYYY-MM-DD | `commit_hash`

### Title of Optimization

**Files Modified:**
- file1.py
- file2.json

**Changes:**
- Description of what was changed

**Results:**
- Before: X.XXs
- After: Y.YYs
- Speedup: Z.Zx

**Notes:**
- Any relevant observations
```
