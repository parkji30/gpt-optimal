# Motivation

We've always heard that low level statically typed languages such as C/C++ and Rust are significantly faster in performance than Python. While top open source libraries such as PyTorch have their heavy lifting written in C++, they are still wrapped around a Python frontend along with any other steps that don't directly call the `model.train()` method.

In this repository, I wanted to investigate and discover the following- "Does training a Large Language Model purely written in Rust show a significant amount of improvement in the training time compared to writing an optimal script in PyTorch?"

The model we will be recreating will be GPT2 as there are extensive resources and documentation around it (OpenAI paper, Karpathy Tutorial).

# Goals
- For this investigation, I will purely be benchmarking the (# of epochs) * (# of steps/iterations) on a target dataset.
    - This consist of loading the data into a batch.
    - Forward passing the batch.
    - Obtaining the loss via the loss/cost function.
    - Backpropagation to obtain the gradients.
    - Step to update the parameters.
- I will be using Burn (Rust package) and the PyTorch (Python Package) libraries to perform this experiment.
- The goal is to see if writing a end-to-end package in Rust can be demonstrate a worthy enough speed up to switch frameworks from PyTorch.

# Assumptions
- Given that the backend of the optimization is different, I do expect to see slight discrepencies on the final evaluation metrics.
- I will not consider development time as part of the total training time since we're just measuring performance here.

---

## Quick Start

### Run the Demo

```bash
cargo run --bin demo
```

This will:
1. Create a tiny GPT-2 model
2. Show tokenization examples
3. Test forward pass
4. Generate sample text (random weights)
5. Evaluate on a demo dataset

### Download Training Data

First, install Python dependencies:
```bash
cd utils
pip install -r requirements.txt
```

Download and prepare datasets:
```bash
# Download demo + WikiText-2 datasets
python download_nemotron.py --datasets demo wikitext

# Or download HelpSteer2 from NVIDIA (requires HuggingFace access)
python download_nemotron.py --datasets helpsteer2

# Download GPT-2 tokenizer files
python prepare_tokenizer.py --use-transformers
```

### Train a Model

```bash
# Train on demo data (quick test)
cargo run --bin train

# Train on WikiText-2
cargo run --bin train -- --data data/wikitext2_train.jsonl --epochs 10 --model-size tiny

# Train with custom settings
cargo run --bin train -- \
    --data data/combined_train.jsonl \
    --model-size small \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 0.0003
```

### Generate Text

```bash
# Basic generation
cargo run --bin generate -- --prompt "The quick brown fox"

# With sampling parameters
cargo run --bin generate -- \
    --prompt "Once upon a time" \
    --max-tokens 100 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9

# Greedy decoding
cargo run --bin generate -- --prompt "Hello world" --greedy
```

## Project Structure

```
├── src/
│   ├── lib.rs              # Library exports
│   ├── main.rs             # CLI entry point
│   ├── model/              # GPT-2 architecture
│   │   ├── mod.rs
│   │   ├── config.rs       # Model configuration
│   │   ├── attention.rs    # Causal self-attention
│   │   ├── transformer.rs  # Transformer blocks + MLP
│   │   └── gpt2.rs         # Full GPT-2 model
│   ├── tokenizer.rs        # Simple tokenizer
│   ├── data/               # Dataset loading
│   │   ├── mod.rs
│   │   ├── dataset.rs      # TextDataset implementation
│   │   └── batcher.rs      # Batch collation
│   ├── training/           # Training pipeline
│   │   ├── mod.rs
│   │   ├── config.rs       # Training configuration
│   │   ├── learner.rs      # Training module wrapper
│   │   └── runner.rs       # Training loop
│   ├── eval/               # Evaluation & generation
│   │   ├── mod.rs
│   │   ├── generator.rs    # Text generation
│   │   └── metrics.rs      # Evaluation metrics
│   ├── visualization/      # Training visualization
│   │   ├── mod.rs
│   │   ├── dashboard.rs    # Real-time dashboard
│   │   └── plots.rs        # ASCII plots
│   └── bin/                # Executable binaries
│       ├── train.rs
│       ├── generate.rs
│       └── demo.rs
├── utils/                  # Python utilities
│   ├── download_nemotron.py    # Dataset download
│   ├── prepare_tokenizer.py    # GPT-2 tokenizer download
│   └── requirements.txt
└── Cargo.toml
```

## Model Configurations

| Size  | d_model | n_heads | n_layers | d_ff  | Parameters |
|-------|---------|---------|----------|-------|------------|
| tiny  | 128     | 2       | 2        | 512   | ~13M       |
| small | 256     | 4       | 4        | 1024  | ~26M       |

## Dependencies

- **[Burn](https://github.com/tracel-ai/burn)** v0.17 - Deep learning framework
- **WGPU** backend - GPU acceleration via WebGPU
- **clap** - Command-line argument parsing
- **serde/serde_json** - Serialization for datasets

## Backends

By default, this uses the WGPU backend which works on most platforms. You can enable different backends:

```toml
# In Cargo.toml
[features]
default = ["wgpu"]
wgpu = ["burn/wgpu"]
cuda = ["burn/cuda"]  # NVIDIA CUDA (requires CUDA toolkit)
tch = ["burn/tch"]    # PyTorch backend (requires libtorch)
```

## Training on Nemotron Data

The NVIDIA Nemotron datasets provide high-quality training data:

1. **HelpSteer2**: 21K human-annotated preference examples
2. **Synthetic Data**: Generated from Nemotron-4-340B

See `utils/download_nemotron.py` for download options.

## License

MIT



