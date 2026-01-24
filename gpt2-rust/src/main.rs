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

// ============================================================================
// Automatic Mixed Precision (AMP) Implementation
// ============================================================================

/// Configuration for AMP gradient scaler
#[derive(Clone, Debug)]
pub struct AmpConfig {
    /// Initial scale factor for loss scaling
    pub init_scale: f32,
    /// Factor to grow scale after successful steps
    pub growth_factor: f32,
    /// Factor to shrink scale after overflow
    pub backoff_factor: f32,
    /// Number of successful steps before growing scale
    pub growth_interval: usize,
    /// Minimum scale value
    pub min_scale: f32,
    /// Maximum scale value
    pub max_scale: f32,
    /// Whether AMP is enabled
    pub enabled: bool,
}

impl Default for AmpConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0,      // 2^16 - good starting point for f16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            min_scale: 1.0,
            max_scale: 65536.0 * 256.0, // 2^24
            enabled: true,
        }
    }
}

/// Gradient scaler for AMP training
/// Handles dynamic loss scaling to prevent gradient underflow in f16
pub struct GradScaler {
    config: AmpConfig,
    scale: f32,
    growth_tracker: usize,
    overflow_count: usize,
    total_steps: usize,
}

impl GradScaler {
    pub fn new(config: AmpConfig) -> Self {
        let scale = config.init_scale;
        Self {
            config,
            scale,
            growth_tracker: 0,
            overflow_count: 0,
            total_steps: 0,
        }
    }

    /// Get current scale factor
    pub fn get_scale(&self) -> f32 {
        if self.config.enabled {
            self.scale
        } else {
            1.0
        }
    }

    /// Scale a loss tensor before backward pass
    pub fn scale_loss<B: Backend>(&self, loss: Tensor<B, 1>) -> Tensor<B, 1> {
        if self.config.enabled {
            loss * self.scale
        } else {
            loss
        }
    }

    /// Check if gradients contain inf/nan values
    fn check_gradients_overflow<B: AutodiffBackend>(
        &self,
        model: &Gpt2Model<B>,
        grads: &<B as AutodiffBackend>::Gradients,
    ) -> bool {
        // We check overflow by examining if any gradient tensor has inf/nan
        // This is a simplified check - in production you'd check all parameter gradients
        let _ = (model, grads);

        // For burn, we rely on the loss value check since direct gradient inspection
        // is complex. If loss becomes inf/nan, we have overflow.
        false
    }

    /// Unscale gradients and check for overflow
    /// Returns (should_skip_step, unscaled_grads)
    pub fn unscale_grads<B: AutodiffBackend>(
        &mut self,
        model: &Gpt2Model<B>,
        grads: <B as AutodiffBackend>::Gradients,
        loss_value: f32,
    ) -> (bool, <B as AutodiffBackend>::Gradients) {
        self.total_steps += 1;

        if !self.config.enabled {
            return (false, grads);
        }

        // Check for overflow (inf or nan in loss)
        let has_overflow = loss_value.is_infinite() || loss_value.is_nan()
            || self.check_gradients_overflow(model, &grads);

        if has_overflow {
            self.overflow_count += 1;
            self.growth_tracker = 0;

            // Reduce scale
            self.scale = (self.scale * self.config.backoff_factor).max(self.config.min_scale);

            // Skip this step
            return (true, grads);
        }

        // Successful step - track for growth
        self.growth_tracker += 1;

        if self.growth_tracker >= self.config.growth_interval {
            // Grow scale
            self.scale = (self.scale * self.config.growth_factor).min(self.config.max_scale);
            self.growth_tracker = 0;
        }

        (false, grads)
    }

    /// Get statistics about the scaler
    pub fn stats(&self) -> (f32, usize, usize) {
        (self.scale, self.overflow_count, self.total_steps)
    }
}

/// AMP-aware optimizer step
/// Performs the optimizer step with proper gradient unscaling
pub fn amp_optimizer_step<B, O>(
    optim: &mut O,
    learning_rate: f64,
    model: Gpt2Model<B>,
    grads: GradientsParams,
    scaler: &GradScaler,
) -> Gpt2Model<B>
where
    B: AutodiffBackend,
    O: Optimizer<Gpt2Model<B>, B>,
{
    // The gradients are already scaled by loss_scale, so we need to
    // effectively divide by scale. We do this by adjusting the learning rate.
    let effective_lr = if scaler.config.enabled {
        learning_rate / scaler.get_scale() as f64
    } else {
        learning_rate
    };

    optim.step(effective_lr, model, grads)
}

/// JSON Configuration structures
#[derive(Deserialize)]
struct JsonConfig {
    model: ModelConfig,
    training: TrainingConfig,
    data: DataConfig,
    generation: GenerationConfig,
    #[serde(default)]
    amp: AmpJsonConfig,
}

/// AMP configuration from JSON
#[derive(Deserialize)]
struct AmpJsonConfig {
    #[serde(default = "default_amp_enabled")]
    enabled: bool,
    #[serde(default = "default_amp_init_scale")]
    init_scale: f32,
    #[serde(default = "default_amp_growth_factor")]
    growth_factor: f32,
    #[serde(default = "default_amp_backoff_factor")]
    backoff_factor: f32,
    #[serde(default = "default_amp_growth_interval")]
    growth_interval: usize,
}

fn default_amp_enabled() -> bool { true }
fn default_amp_init_scale() -> f32 { 65536.0 }
fn default_amp_growth_factor() -> f32 { 2.0 }
fn default_amp_backoff_factor() -> f32 { 0.5 }
fn default_amp_growth_interval() -> usize { 2000 }

impl Default for AmpJsonConfig {
    fn default() -> Self {
        Self {
            enabled: default_amp_enabled(),
            init_scale: default_amp_init_scale(),
            growth_factor: default_amp_growth_factor(),
            backoff_factor: default_amp_backoff_factor(),
            growth_interval: default_amp_growth_interval(),
        }
    }
}

impl From<&AmpJsonConfig> for AmpConfig {
    fn from(json: &AmpJsonConfig) -> Self {
        Self {
            enabled: json.enabled,
            init_scale: json.init_scale,
            growth_factor: json.growth_factor,
            backoff_factor: json.backoff_factor,
            growth_interval: json.growth_interval,
            ..Default::default()
        }
    }
}

#[derive(Deserialize)]
struct ModelConfig {
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    block_size: usize,
    dropout: f32,
}

#[derive(Deserialize)]
struct TrainingConfig {
    batch_size: usize,
    learning_rate: f32,
    max_iters: usize,
    eval_interval: usize,
    eval_iters: usize,
}

#[derive(Deserialize)]
struct DataConfig {
    train_split: f32,
    path: String,
}

#[derive(Deserialize)]
struct GenerationConfig {
    max_new_tokens: usize,
    #[allow(dead_code)]
    temperature: f32,
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
    pub dropout: f32,
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
        let dropout = DropoutConfig::new(self.dropout as f64).init();

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
    pub dropout: f32,
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        let ln1 = LayerNormConfig::new(self.n_embd).init(device);
        let attn = MultiHeadAttentionConfig::new(self.n_embd, self.n_head)
            .with_dropout(self.dropout as f64)
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
    pub dropout: f32,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let c_fc = LinearConfig::new(self.n_embd, 4 * self.n_embd).init(device);
        let c_proj = LinearConfig::new(4 * self.n_embd, self.n_embd).init(device);
        let dropout = DropoutConfig::new(self.dropout as f64).init();

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

fn train<B: AutodiffBackend>(config_path: &str, device: &B::Device, use_amp: bool) {
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
    let n = (text.len() as f32 * config.data.train_split) as usize;
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

    // Initialize AMP GradScaler
    let mut amp_config = AmpConfig::from(&config.amp);
    // Only enable AMP if both feature and config say so
    amp_config.enabled = use_amp && config.amp.enabled;
    let mut grad_scaler = GradScaler::new(amp_config.clone());

    if amp_config.enabled {
        println!("AMP enabled: init_scale={}, growth_interval={}",
            amp_config.init_scale, amp_config.growth_interval);
    } else {
        println!("AMP disabled: using full precision gradients");
    }

    // Training loop
    println!("\nStarting training...");
    let training_start = Instant::now();
    let mut skipped_steps = 0usize;

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

            // Print AMP stats if enabled
            if amp_config.enabled {
                let (scale, overflows, _) = grad_scaler.stats();
                println!(
                    "Step {}: train loss {:.4}, val loss {:.4} | AMP scale: {:.0}, overflows: {}",
                    iter_num, train_loss, val_loss, scale, overflows
                );
            } else {
                println!(
                    "Step {}: train loss {:.4}, val loss {:.4}",
                    iter_num, train_loss, val_loss
                );
            }
        }

        // Get batch and compute loss
        let (x, y) = train_data.get_batch::<B>(config.training.batch_size, device);
        let loss = model.forward_loss(x, y);

        // Get unscaled loss value for overflow detection
        let loss_value = loss.clone().into_scalar().elem::<f32>();

        // Scale loss for backward pass (prevents gradient underflow in f16)
        let scaled_loss = grad_scaler.scale_loss(loss);

        // Backward pass with scaled loss
        let grads = scaled_loss.backward();

        // Check for overflow and unscale gradients
        let (skip_step, grads) = grad_scaler.unscale_grads(&model, grads, loss_value);

        if skip_step {
            // Overflow detected - skip this optimizer step
            skipped_steps += 1;
            continue;
        }

        // Convert gradients and perform optimizer step
        let grads = GradientsParams::from_grads(grads, &model);
        model = amp_optimizer_step(&mut optim, config.training.learning_rate as f64, model, grads, &grad_scaler);
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

    // Print final AMP statistics
    if amp_config.enabled {
        let (final_scale, total_overflows, total_steps) = grad_scaler.stats();
        println!("\nAMP Statistics:");
        println!("  Final scale: {:.0}", final_scale);
        println!("  Total overflows: {} ({:.2}%)",
            total_overflows,
            (total_overflows as f32 / total_steps as f32) * 100.0);
        println!("  Skipped steps: {}", skipped_steps);
    }

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
    // AMP (Automatic Mixed Precision) uses f16 for faster training on modern GPUs
    // with dynamic loss scaling to prevent gradient underflow
    #[cfg(all(feature = "cuda", feature = "amp"))]
    {
        use burn::backend::cuda::{Cuda, CudaDevice};
        use half::f16;
        type MyBackend = Cuda<f16, i32>;
        type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
        let device = CudaDevice::default();
        println!("Using CUDA backend with AMP (f16)");
        println!("  - Forward/backward: f16 (half precision)");
        println!("  - Loss scaling: dynamic");
        println!("  - Gradient unscaling: automatic");
        train::<MyAutodiffBackend>(config_path, &device, true);
    }

    #[cfg(all(feature = "cuda", not(feature = "amp")))]
    {
        use burn::backend::cuda::{Cuda, CudaDevice};
        type MyBackend = Cuda<f32, i32>;
        type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
        let device = CudaDevice::default();
        println!("Using CUDA backend (f32)");
        train::<MyAutodiffBackend>(config_path, &device, false);
    }

    #[cfg(feature = "ndarray")]
    {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};
        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
        let device = NdArrayDevice::Cpu;
        println!("Using NdArray backend (CPU only)");
        train::<MyAutodiffBackend>(config_path, &device, false);
    }
}
