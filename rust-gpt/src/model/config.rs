use burn::config::Config;

/// Configuration for GPT-2 model
#[derive(Config, Debug)]
pub struct GPT2Config {
    /// Vocabulary size
    #[config(default = 50257)]
    pub vocab_size: usize,
    
    /// Maximum sequence length
    #[config(default = 1024)]
    pub max_seq_len: usize,
    
    /// Embedding dimension
    #[config(default = 768)]
    pub d_model: usize,
    
    /// Number of attention heads
    #[config(default = 12)]
    pub n_heads: usize,
    
    /// Number of transformer layers
    #[config(default = 12)]
    pub n_layers: usize,
    
    /// Dimension of feed-forward network
    #[config(default = 3072)]
    pub d_ff: usize,
    
    /// Dropout probability
    #[config(default = 0.1)]
    pub dropout: f64,
    
    /// Layer normalization epsilon
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

impl GPT2Config {
    /// Create a small GPT-2 config for testing/training from scratch
    pub fn small() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 256,
            d_model: 256,
            n_heads: 4,
            n_layers: 4,
            d_ff: 1024,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
    
    /// Create a tiny GPT-2 config for quick experiments
    pub fn tiny() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 128,
            d_model: 128,
            n_heads: 2,
            n_layers: 2,
            d_ff: 512,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}


