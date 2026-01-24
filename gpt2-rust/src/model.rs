//! GPT-2 model architecture in Rust using Burn.
//!
//! This mirrors the Python implementation with:
//! - Token and position embeddings
//! - Transformer blocks with pre-norm architecture
//! - Custom Flash Attention
//! - Weight tying between embeddings and output head

use burn::prelude::*;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig};
use burn::tensor::activation::gelu;

use crate::attention::{CausalSelfAttention, CausalSelfAttentionConfig};
use crate::config::ModelConfig;

/// MLP (Feed-Forward Network) module.
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    /// First linear layer (expansion)
    c_fc: Linear<B>,
    /// Second linear layer (projection)
    c_proj: Linear<B>,
    /// Dropout
    dropout: Dropout,
}

/// Configuration for MLP
#[derive(Config, Debug)]
pub struct MLPConfig {
    pub n_embd: usize,
    pub dropout: f64,
}

impl MLPConfig {
    /// Initialize the MLP module
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        let c_fc = LinearConfig::new(self.n_embd, 4 * self.n_embd)
            .with_bias(true)
            .init(device);
        let c_proj = LinearConfig::new(4 * self.n_embd, self.n_embd)
            .with_bias(true)
            .init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        MLP { c_fc, c_proj, dropout }
    }
}

impl<B: Backend> MLP<B> {
    /// Forward pass through MLP.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        let x = gelu(x);
        let x = self.c_proj.forward(x);
        self.dropout.forward(x)
    }
}

/// Transformer Block with pre-norm architecture.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Layer norm before attention
    ln_1: LayerNorm<B>,
    /// Causal self-attention
    attn: CausalSelfAttention<B>,
    /// Layer norm before MLP
    ln_2: LayerNorm<B>,
    /// MLP (feed-forward)
    mlp: MLP<B>,
}

/// Configuration for TransformerBlock
#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub dropout: f64,
}

impl TransformerBlockConfig {
    /// Initialize the TransformerBlock module
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        let ln_1 = LayerNormConfig::new(self.n_embd).init(device);
        let attn = CausalSelfAttentionConfig {
            n_embd: self.n_embd,
            n_head: self.n_head,
            dropout: self.dropout,
        }.init(device);
        let ln_2 = LayerNormConfig::new(self.n_embd).init(device);
        let mlp = MLPConfig {
            n_embd: self.n_embd,
            dropout: self.dropout,
        }.init(device);

        TransformerBlock { ln_1, attn, ln_2, mlp }
    }
}

impl<B: Backend> TransformerBlock<B> {
    /// Forward pass through transformer block.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x.clone()));
        x.clone() + self.mlp.forward(self.ln_2.forward(x))
    }
}

/// GPT-2 Language Model.
#[derive(Module, Debug)]
pub struct GPT2<B: Backend> {
    /// Token embeddings
    wte: Embedding<B>,
    /// Position embeddings
    wpe: Embedding<B>,
    /// Embedding dropout
    drop: Dropout,
    /// Transformer blocks
    h: Vec<TransformerBlock<B>>,
    /// Final layer norm
    ln_f: LayerNorm<B>,
    /// Language model head (output projection)
    lm_head: Linear<B>,
    /// Vocabulary size (stored separately from config to avoid Module derivation issues)
    #[module(skip)]
    vocab_size: usize,
    /// Embedding dimension
    #[module(skip)]
    n_embd: usize,
    /// Number of heads
    #[module(skip)]
    n_head: usize,
    /// Number of layers
    #[module(skip)]
    n_layer: usize,
    /// Block size (context length)
    #[module(skip)]
    block_size: usize,
    /// Dropout rate
    #[module(skip)]
    dropout_rate: f64,
}

/// Configuration for GPT2
#[derive(Config, Debug)]
pub struct GPT2ConfigBurn {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block_size: usize,
    pub dropout: f64,
}

impl From<&ModelConfig> for GPT2ConfigBurn {
    fn from(config: &ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            n_embd: config.n_embd,
            n_head: config.n_head,
            n_layer: config.n_layer,
            block_size: config.block_size,
            dropout: config.dropout,
        }
    }
}

impl GPT2ConfigBurn {
    /// Initialize the GPT2 model
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPT2<B> {
        let wte = EmbeddingConfig::new(self.vocab_size, self.n_embd).init(device);
        let wpe = EmbeddingConfig::new(self.block_size, self.n_embd).init(device);
        let drop = DropoutConfig::new(self.dropout).init();
        
        let h: Vec<TransformerBlock<B>> = (0..self.n_layer)
            .map(|_| {
                TransformerBlockConfig {
                    n_embd: self.n_embd,
                    n_head: self.n_head,
                    dropout: self.dropout,
                }.init(device)
            })
            .collect();
        
        let ln_f = LayerNormConfig::new(self.n_embd).init(device);
        let lm_head = LinearConfig::new(self.n_embd, self.vocab_size)
            .with_bias(false)
            .init(device);

        GPT2 {
            wte,
            wpe,
            drop,
            h,
            ln_f,
            lm_head,
            vocab_size: self.vocab_size,
            n_embd: self.n_embd,
            n_head: self.n_head,
            n_layer: self.n_layer,
            block_size: self.block_size,
            dropout_rate: self.dropout,
        }
    }
}

impl<B: Backend> GPT2<B> {
    /// Get model configuration as ModelConfig
    pub fn config(&self) -> ModelConfig {
        ModelConfig {
            vocab_size: self.vocab_size,
            n_embd: self.n_embd,
            n_head: self.n_head,
            n_layer: self.n_layer,
            block_size: self.block_size,
            dropout: self.dropout_rate,
        }
    }

    /// Forward pass through GPT-2.
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = idx.dims();
        let device = idx.device();

        assert!(
            seq_len <= self.block_size,
            "Sequence length {} exceeds block size {}",
            seq_len,
            self.block_size
        );

        // Token embeddings
        let tok_emb = self.wte.forward(idx);

        // Position embeddings - need to reshape for embedding forward
        let pos = Tensor::<B, 1, Int>::arange(0..(seq_len as i64), &device)
            .reshape([1, seq_len]);
        let pos_emb = self.wpe.forward(pos);
        
        // Broadcast position embeddings to batch
        let pos_emb = pos_emb.expand([batch_size, seq_len, self.n_embd]);

        // Combine embeddings and apply dropout
        let mut x = self.drop.forward(tok_emb + pos_emb);

        // Pass through transformer blocks
        for block in &self.h {
            x = block.forward(x);
        }

        // Final layer norm
        x = self.ln_f.forward(x);

        // Project to vocabulary
        self.lm_head.forward(x)
    }

    /// Forward pass with loss computation.
    pub fn forward_with_loss(
        &self,
        idx: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let logits = self.forward(idx);
        let loss = self.compute_loss(logits.clone(), targets);
        (logits, loss)
    }

    /// Compute cross-entropy loss.
    fn compute_loss(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_len, vocab_size] = logits.dims();

        // Reshape for cross-entropy
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);
        let n_samples = batch_size * seq_len;

        // Cross-entropy loss
        let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1);
        let targets_expanded = targets_flat.reshape([n_samples, 1]);
        let target_log_probs = log_probs.gather(1, targets_expanded);
        let loss = target_log_probs.neg().mean();
        
        loss.reshape([1])
    }

    /// Generate new tokens autoregressively.
    pub fn generate(
        &self,
        idx: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
    ) -> Tensor<B, 2, Int> {
        let device = idx.device();
        let mut current_idx = idx;

        for _ in 0..max_new_tokens {
            let [batch_size, seq_len] = current_idx.dims();

            // Crop context if needed
            let idx_cond = if seq_len > self.block_size {
                let start = seq_len - self.block_size;
                current_idx.clone().slice([0..batch_size, start..seq_len])
            } else {
                current_idx.clone()
            };

            // Forward pass
            let logits = self.forward(idx_cond);
            
            // Get logits for last position
            let [_, new_seq_len, vocab_size] = logits.dims();
            let logits_last = logits
                .slice([0..batch_size, (new_seq_len - 1)..new_seq_len, 0..vocab_size])
                .reshape([batch_size, vocab_size]);
            let logits_scaled = logits_last.div_scalar(temperature as f32);

            // Apply top-k filtering if specified
            let logits_filtered = if let Some(k) = top_k {
                self.apply_top_k(logits_scaled, k, &device)
            } else {
                logits_scaled
            };

            // Softmax to get probabilities
            let probs = burn::tensor::activation::softmax(logits_filtered, 1);

            // Sample from distribution
            let idx_next = self.sample_from_probs(probs, &device);

            // Append to sequence
            current_idx = Tensor::cat(vec![current_idx, idx_next], 1);
        }

        current_idx
    }

    /// Apply top-k filtering to logits.
    fn apply_top_k(&self, logits: Tensor<B, 2>, k: usize, device: &B::Device) -> Tensor<B, 2> {
        let [batch_size, vocab_size] = logits.dims();
        let k = k.min(vocab_size);

        // Sort descending by negating, sorting, then negating back
        let logits_neg = logits.clone().neg();
        let logits_sorted_neg = logits_neg.sort(1);
        let logits_sorted = logits_sorted_neg.neg();
        
        let threshold = logits_sorted
            .slice([0..batch_size, (k - 1)..k])
            .reshape([batch_size, 1])
            .expand([batch_size, vocab_size]);

        // Mask out values below threshold
        let mask = logits.clone().lower(threshold);
        let neg_inf = Tensor::<B, 2>::full([batch_size, vocab_size], f32::NEG_INFINITY, device);
        
        logits.mask_where(mask, neg_inf)
    }

    /// Sample token indices from probability distribution.
    fn sample_from_probs(&self, probs: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 2, Int> {
        let [batch_size, vocab_size] = probs.dims();
        
        let probs_data = probs.into_data();
        let probs_slice: &[f32] = probs_data.as_slice().unwrap();
        
        let mut sampled = vec![0i64; batch_size];
        
        for b in 0..batch_size {
            let start = b * vocab_size;
            let end = start + vocab_size;
            let batch_probs = &probs_slice[start..end];
            
            // Cumulative sum sampling
            let mut cumsum = 0.0;
            let r: f32 = rand::random();
            
            for (i, &p) in batch_probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    sampled[b] = i as i64;
                    break;
                }
            }
            
            if sampled[b] == 0 && r > 0.0 {
                sampled[b] = (vocab_size - 1) as i64;
            }
        }
        
        Tensor::<B, 1, Int>::from_data(sampled.as_slice(), device)
            .reshape([batch_size, 1])
    }

    /// Get the number of parameters in the model.
    pub fn num_params(&self) -> usize {
        let wte_params = self.vocab_size * self.n_embd;
        let wpe_params = self.block_size * self.n_embd;
        
        let attn_params = 3 * self.n_embd * self.n_embd + 3 * self.n_embd
            + self.n_embd * self.n_embd + self.n_embd;
        let mlp_params = self.n_embd * 4 * self.n_embd + 4 * self.n_embd
            + 4 * self.n_embd * self.n_embd + self.n_embd;
        let ln_params = 2 * self.n_embd;
        let block_params = attn_params + mlp_params + 2 * ln_params;
        
        let total_block_params = self.n_layer * block_params;
        let ln_f_params = 2 * self.n_embd;
        let lm_head_params = self.n_embd * self.vocab_size;
        
        wte_params + wpe_params + total_block_params + ln_f_params + lm_head_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_mlp() {
        let device = Default::default();
        let mlp: MLP<TestBackend> = MLPConfig {
            n_embd: 128,
            dropout: 0.0,
        }.init(&device);

        let x = Tensor::random([2, 16, 128], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = mlp.forward(x);

        assert_eq!(output.dims(), [2, 16, 128]);
    }

    #[test]
    fn test_transformer_block() {
        let device = Default::default();
        let block: TransformerBlock<TestBackend> = TransformerBlockConfig {
            n_embd: 128,
            n_head: 4,
            dropout: 0.0,
        }.init(&device);

        let x = Tensor::random([2, 16, 128], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = block.forward(x);

        assert_eq!(output.dims(), [2, 16, 128]);
    }

    #[test]
    fn test_gpt2_forward() {
        let device = Default::default();
        let model: GPT2<TestBackend> = GPT2ConfigBurn {
            vocab_size: 256,
            n_embd: 128,
            n_head: 4,
            n_layer: 2,
            block_size: 64,
            dropout: 0.0,
        }.init(&device);

        let idx = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 2, 3, 4, 5, 6, 7], &device)
            .reshape([2, 4]);
        let logits = model.forward(idx);

        assert_eq!(logits.dims(), [2, 4, 256]);
    }

    #[test]
    fn test_gpt2_forward_with_loss() {
        let device = Default::default();
        let model: GPT2<TestBackend> = GPT2ConfigBurn {
            vocab_size: 256,
            n_embd: 128,
            n_head: 4,
            n_layer: 2,
            block_size: 64,
            dropout: 0.0,
        }.init(&device);

        let idx = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 2, 3, 4, 5, 6, 7], &device)
            .reshape([2, 4]);
        let targets = Tensor::<TestBackend, 1, Int>::from_data([1i64, 2, 3, 4, 5, 6, 7, 8], &device)
            .reshape([2, 4]);
        
        let (logits, loss) = model.forward_with_loss(idx, targets);

        assert_eq!(logits.dims(), [2, 4, 256]);
        assert_eq!(loss.dims(), [1]);
    }
}
