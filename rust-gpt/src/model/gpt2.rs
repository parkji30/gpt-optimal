use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor},
};

use super::{config::GPT2Config, transformer::TransformerBlock};

/// GPT-2 Language Model
#[derive(Module, Debug)]
pub struct GPT2<B: Backend> {
    /// Token embeddings
    wte: Embedding<B>,
    /// Position embeddings
    wpe: Embedding<B>,
    /// Dropout
    drop: Dropout,
    /// Transformer blocks
    blocks: Vec<TransformerBlock<B>>,
    /// Final layer normalization
    ln_f: LayerNorm<B>,
    /// Language model head (tied with wte in original GPT-2, but separate here for simplicity)
    lm_head: Linear<B>,
    /// Vocab size (stored separately since Config is not a Module)
    vocab_size: usize,
    /// Max sequence length
    max_seq_len: usize,
    /// Embedding dimension
    d_model: usize,
}

impl<B: Backend> GPT2<B> {
    pub fn new(config: &GPT2Config, device: &B::Device) -> Self {
        // Token embeddings
        let wte = EmbeddingConfig::new(config.vocab_size, config.d_model).init(device);
        
        // Position embeddings
        let wpe = EmbeddingConfig::new(config.max_seq_len, config.d_model).init(device);
        
        // Dropout
        let drop = DropoutConfig::new(config.dropout).init();
        
        // Transformer blocks
        let blocks: Vec<TransformerBlock<B>> = (0..config.n_layers)
            .map(|_| {
                TransformerBlock::new(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.layer_norm_eps,
                    device,
                )
            })
            .collect();
        
        // Final layer norm
        let ln_f = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        
        // LM head
        let lm_head = LinearConfig::new(config.d_model, config.vocab_size)
            .with_bias(false)
            .init(device);
        
        Self {
            wte,
            wpe,
            drop,
            blocks,
            ln_f,
            lm_head,
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            d_model: config.d_model,
        }
    }
    
    /// Forward pass returning logits
    /// Input: [batch, seq_len] token indices
    /// Output: [batch, seq_len, vocab_size] logits
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = input_ids.dims();
        let device = input_ids.device();
        
        // Create position indices
        let positions: Tensor<B, 2, Int> = Tensor::arange(0..seq_len as i64, &device)
            .reshape([1, seq_len])
            .repeat_dim(0, batch);
        
        // Get embeddings
        let tok_emb = self.wte.forward(input_ids);
        let pos_emb = self.wpe.forward(positions);
        
        // Combine embeddings
        let mut x = self.drop.forward(tok_emb + pos_emb);
        
        // Pass through transformer blocks
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        // Final layer norm and project to vocabulary
        let x = self.ln_f.forward(x);
        self.lm_head.forward(x)
    }
    
    /// Calculate cross-entropy loss for language modeling
    /// Input: [batch, seq_len] token indices
    /// Targets: [batch, seq_len] target token indices (shifted by 1)
    pub fn forward_loss(&self, input_ids: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let logits = self.forward(input_ids);
        let [batch, seq_len, vocab_size] = logits.dims();
        
        // Reshape for cross-entropy: [batch * seq_len, vocab_size]
        let logits = logits.reshape([batch * seq_len, vocab_size]);
        let targets = targets.reshape([batch * seq_len]);
        
        // Cross-entropy loss
        burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, targets)
    }
    
    /// Generate text autoregressively
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f64,
    ) -> Tensor<B, 2, Int> {
        let mut current_ids = input_ids;
        
        for _ in 0..max_new_tokens {
            let [_batch, seq_len] = current_ids.dims();
            let device = current_ids.device();
            
            // Truncate if exceeding max_seq_len
            let context_ids = if seq_len > self.max_seq_len {
                current_ids.clone().slice([0..1, (seq_len - self.max_seq_len)..seq_len])
            } else {
                current_ids.clone()
            };
            
            // Get logits for the last position
            let logits = self.forward(context_ids);
            let [batch, seq_len_out, vocab_size] = logits.dims();
            
            // Get last token logits: [batch, vocab_size]
            let last_logits = logits.slice([0..batch, (seq_len_out - 1)..seq_len_out, 0..vocab_size])
                .reshape([batch, vocab_size]);
            
            // Apply temperature
            let scaled_logits = last_logits / temperature;
            
            // Sample from softmax distribution (greedy for simplicity)
            let probs = burn::tensor::activation::softmax(scaled_logits, 1);
            let next_token = probs.argmax(1).reshape([batch, 1]);
            
            // Append to sequence
            current_ids = Tensor::cat(vec![current_ids, next_token], 1);
        }
        
        current_ids
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get max sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    
    /// Get embedding dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

