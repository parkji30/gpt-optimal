use burn::module::{AutodiffModule, Module};
use burn::nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    loss::CrossEntropyLossConfig,
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig,
    Linear, LinearConfig,
};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::backend::AutodiffBackend;
use rand::Rng;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// JSON Configuration structures
#[derive(Deserialize)]
struct JsonConfig {
    model: ModelConfig,
    training: TrainingConfig,
    data: DataConfig,
    generation: GenerationConfig,
}

#[derive(Deserialize)]
struct ModelConfig {
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    block_size: usize,
    dropout: f64,
}

#[derive(Deserialize)]
struct TrainingConfig {
    batch_size: usize,
    learning_rate: f64,
    max_iters: usize,
    eval_interval: usize,
    eval_iters: usize,
}

#[derive(Deserialize)]
struct DataConfig {
    train_split: f64,
    path: String,
}

#[derive(Deserialize)]
struct GenerationConfig {
    max_new_tokens: usize,
    #[allow(dead_code)]
    temperature: f64,
}

/// GPT-2 Configuration
#[derive(Config, Debug)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block_size: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl Gpt2Config {
    pub fn from_json(model_cfg: &ModelConfig) -> Self {
        Self::new(
            model_cfg.vocab_size,
            model_cfg.n_embd,
            model_cfg.n_head,
            model_cfg.n_layer,
            model_cfg.block_size,
        )
        .with_dropout(model_cfg.dropout)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2Model<B> {
        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.n_embd).init(device);
        let position_embedding = EmbeddingConfig::new(self.block_size, self.n_embd).init(device);

        let blocks: Vec<_> = (0..self.n_layer)
            .map(|_| {
                TransformerBlockConfig::new(self.n_embd, self.n_head)
                    .with_dropout(self.dropout)
                    .init(device)
            })
            .collect();

        let ln_f = LayerNormConfig::new(self.n_embd).init(device);
        let lm_head = LinearConfig::new(self.n_embd, self.vocab_size)
            .with_bias(false)
            .init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        Gpt2Model {
            token_embedding,
            position_embedding,
            blocks,
            ln_f,
            lm_head,
            dropout,
            block_size: self.block_size,
            vocab_size: self.vocab_size,
        }
    }
}

/// GPT-2 Model
#[derive(Module, Debug)]
pub struct Gpt2Model<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<TransformerBlock<B>>,
    ln_f: LayerNorm<B>,
    lm_head: Linear<B>,
    dropout: Dropout,
    block_size: usize,
    vocab_size: usize,
}

impl<B: Backend> Gpt2Model<B> {
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = &idx.device();
        let [_batch, seq_len] = idx.dims();

        // Token embeddings
        let tok_emb = self.token_embedding.forward(idx);

        // Position embeddings
        let positions = Tensor::arange(0..seq_len as i64, device).unsqueeze::<2>();
        let pos_emb = self.position_embedding.forward(positions);

        // Combine embeddings
        let mut x = tok_emb + pos_emb;
        x = self.dropout.forward(x);

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Final layer norm and projection
        x = self.ln_f.forward(x);
        self.lm_head.forward(x)
    }

    pub fn forward_loss(&self, idx: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let logits = self.forward(idx);
        let [batch, seq, vocab] = logits.dims();

        // Reshape for cross entropy: [batch * seq, vocab]
        let logits = logits.reshape([batch * seq, vocab]);
        // Reshape targets: [batch * seq]
        let targets = targets.reshape([batch * seq]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, targets);

        loss.unsqueeze()
    }

    pub fn generate(&self, idx: Tensor<B, 2, Int>, max_new_tokens: usize) -> Tensor<B, 2, Int> {
        let mut idx = idx;

        for _ in 0..max_new_tokens {
            let [batch_size, seq_len] = idx.dims();

            // Crop to block_size if needed
            let idx_cond = if seq_len > self.block_size {
                idx.clone()
                    .slice([0..batch_size, (seq_len - self.block_size)..seq_len])
            } else {
                idx.clone()
            };

            // Get predictions
            let logits = self.forward(idx_cond);
            let [batch, seq, vocab] = logits.dims();

            // Focus on last time step
            let last_logits = logits.narrow(1, seq - 1, 1);
            let logits: Tensor<B, 2> = last_logits.reshape([batch, vocab]);

            // Apply softmax
            let probs = softmax(logits, 1);

            // Sample from distribution (greedy - argmax returns [batch, 1])
            let idx_next: Tensor<B, 2, Int> = probs.argmax(1);

            // Append to sequence
            idx = Tensor::cat(vec![idx, idx_next], 1);
        }

        idx
    }
}

/// Transformer Block Configuration
#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    pub n_embd: usize,
    pub n_head: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        let ln1 = LayerNormConfig::new(self.n_embd).init(device);
        let attn = MultiHeadAttentionConfig::new(self.n_embd, self.n_head)
            .with_dropout(self.dropout)
            .init(device);
        let ln2 = LayerNormConfig::new(self.n_embd).init(device);
        let mlp = MlpConfig::new(self.n_embd).with_dropout(self.dropout).init(device);

        TransformerBlock { ln1, attn, ln2, mlp }
    }
}

/// Transformer Block
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    ln1: LayerNorm<B>,
    attn: MultiHeadAttention<B>,
    ln2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let [_batch, seq_len, _dim] = x.dims();

        // Self-attention with residual
        let ln_x = self.ln1.forward(x.clone());

        // Create causal mask: true for positions to MASK (upper triangle)
        let causal_mask: Tensor<B, 3, Bool> = Tensor::<B, 2>::ones([seq_len, seq_len], &device)
            .triu(1)  // Upper triangular (excluding diagonal) = positions to mask
            .equal_elem(1.0)
            .unsqueeze::<3>();  // Add batch dimension

        let attn_input = MhaInput::self_attn(ln_x).mask_attn(causal_mask);
        let attn_out = self.attn.forward(attn_input).context;
        let x = x + attn_out;

        // MLP with residual
        let ln_x = self.ln2.forward(x.clone());
        x + self.mlp.forward(ln_x)
    }
}

/// MLP Configuration
#[derive(Config, Debug)]
pub struct MlpConfig {
    pub n_embd: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let c_fc = LinearConfig::new(self.n_embd, 4 * self.n_embd).init(device);
        let c_proj = LinearConfig::new(4 * self.n_embd, self.n_embd).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        Mlp {
            c_fc,
            c_proj,
            gelu: Gelu::new(),
            dropout,
        }
    }
}

/// MLP (Feed-Forward Network)
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    c_fc: Linear<B>,
    c_proj: Linear<B>,
    gelu: Gelu,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        let x = self.gelu.forward(x);
        let x = self.c_proj.forward(x);
        self.dropout.forward(x)
    }
}

/// Character-level dataset
struct CharDataset {
    data: Vec<i32>,
    block_size: usize,
}

impl CharDataset {
    fn new(text: &str, block_size: usize) -> Self {
        let data: Vec<i32> = text.bytes().map(|b| b as i32).collect();
        Self { data, block_size }
    }

    fn len(&self) -> usize {
        self.data.len().saturating_sub(self.block_size)
    }

    fn get_batch<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let mut rng = rand::thread_rng();
        let max_idx = self.len();

        let mut x_data = Vec::with_capacity(batch_size * self.block_size);
        let mut y_data = Vec::with_capacity(batch_size * self.block_size);

        for _ in 0..batch_size {
            let i = rng.gen_range(0..max_idx);
            x_data.extend_from_slice(&self.data[i..i + self.block_size]);
            y_data.extend_from_slice(&self.data[i + 1..i + self.block_size + 1]);
        }

        let x = Tensor::from_data(
            TensorData::new(x_data, [batch_size, self.block_size]),
            device,
        );
        let y = Tensor::from_data(
            TensorData::new(y_data, [batch_size, self.block_size]),
            device,
        );

        (x, y)
    }
}

fn load_config(config_path: &str) -> JsonConfig {
    let content = fs::read_to_string(config_path).expect("Failed to read config file");
    serde_json::from_str(&content).expect("Failed to parse config JSON")
}

fn estimate_loss<B: Backend>(
    model: &Gpt2Model<B>,
    train_data: &CharDataset,
    val_data: &CharDataset,
    eval_iters: usize,
    batch_size: usize,
    device: &B::Device,
) -> (f32, f32) {
    let mut train_loss = 0.0;
    let mut val_loss = 0.0;

    for _ in 0..eval_iters {
        let (x, y) = train_data.get_batch::<B>(batch_size, device);
        let loss = model.forward_loss(x, y);
        train_loss += loss.into_scalar().elem::<f32>();

        let (x, y) = val_data.get_batch::<B>(batch_size, device);
        let loss = model.forward_loss(x, y);
        val_loss += loss.into_scalar().elem::<f32>();
    }

    (
        train_loss / eval_iters as f32,
        val_loss / eval_iters as f32,
    )
}

fn decode(tokens: &[i32]) -> String {
    tokens.iter().map(|&t| t as u8 as char).collect()
}

fn train<B: AutodiffBackend>(config_path: &str, device: &B::Device) {
    // Load configuration
    let config = load_config(config_path);
    let model_config = Gpt2Config::from_json(&config.model);

    // Load data
    let config_dir = Path::new(config_path).parent().unwrap();
    let data_path = config_dir.join(&config.data.path);
    println!("Loading data from: {:?}", data_path);

    let text = fs::read_to_string(&data_path).expect("Failed to read data file");
    println!("Loaded {} characters", text.len());

    // Train/val split
    let n = (text.len() as f64 * config.data.train_split) as usize;
    let train_data = CharDataset::new(&text[..n], model_config.block_size);
    let val_data = CharDataset::new(&text[n..], model_config.block_size);
    println!(
        "Train size: {}, Val size: {}",
        train_data.len(),
        val_data.len()
    );

    // Initialize model
    let mut model: Gpt2Model<B> = model_config.init(device);
    let n_params: usize = model.clone().num_params();
    println!("Model parameters: {}", n_params);

    // Optimizer
    let mut optim = AdamConfig::new().init();

    // Training loop
    println!("\nStarting training...");
    let training_start = Instant::now();
    for iter_num in 0..config.training.max_iters {
        // Evaluate periodically
        if iter_num % config.training.eval_interval == 0 {
            let inner_model: Gpt2Model<B::InnerBackend> = model.clone().valid();
            let (train_loss, val_loss) = estimate_loss(
                &inner_model,
                &train_data,
                &val_data,
                config.training.eval_iters,
                config.training.batch_size,
                device,
            );
            println!(
                "Step {}: train loss {:.4}, val loss {:.4}",
                iter_num, train_loss, val_loss
            );
        }

        // Get batch and compute loss
        let (x, y) = train_data.get_batch::<B>(config.training.batch_size, device);
        let loss = model.forward_loss(x, y);

        // Backward pass
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.training.learning_rate, model, grads);
    }
    let training_duration = training_start.elapsed();

    // Final evaluation
    let inner_model: Gpt2Model<B::InnerBackend> = model.clone().valid();
    let (train_loss, val_loss) = estimate_loss(
        &inner_model,
        &train_data,
        &val_data,
        config.training.eval_iters,
        config.training.batch_size,
        device,
    );
    println!(
        "\nFinal: train loss {:.4}, val loss {:.4}",
        train_loss, val_loss
    );

    println!("\nTraining complete!");
    println!("Total training time: {:.2}s", training_duration.as_secs_f64());

    // Generate sample
    println!("\nGenerating sample...");
    // Start with a newline character (ASCII 10)
    let start_token: i32 = 10;
    let context: Tensor<B::InnerBackend, 2, Int> = Tensor::from_data(
        TensorData::new(vec![start_token], [1, 1]),
        device,
    );
    let generated = inner_model.generate(context, config.generation.max_new_tokens);
    // Convert tensor to vec - use to_vec for proper GPU->CPU sync
    let [_batch, seq_len] = generated.dims();
    let flattened: Tensor<B::InnerBackend, 1, Int> = generated.reshape([seq_len]);
    let data = flattened.into_data();
    let tokens: Vec<i32> = data.to_vec::<i32>().expect("Failed to convert tensor data to i32 vec");
    let output_text = decode(&tokens);
    println!("{}", "=".repeat(50));
    println!("{}", output_text);
    println!("{}", "=".repeat(50));
}

fn main() {
    // Get config path relative to executable
    let config_path = "../hyperparams/config.json";

    // Select backend based on feature flags
    #[cfg(feature = "cuda")]
    {
        use burn::backend::cuda::{Cuda, CudaDevice};
        type MyBackend = Cuda<f32, i32>;
        type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
        let device = CudaDevice::default();
        println!("Using CUDA backend");
        train::<MyAutodiffBackend>(config_path, &device);
    }

    #[cfg(feature = "ndarray")]
    {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};
        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
        let device = NdArrayDevice::Cpu;
        println!("Using NdArray backend (CPU only)");
        train::<MyAutodiffBackend>(config_path, &device);
    }
}
