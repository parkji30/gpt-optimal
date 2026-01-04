use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

/// Causal Multi-Head Self-Attention
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    /// Query, Key, Value projections combined
    c_attn: Linear<B>,
    /// Output projection
    c_proj: Linear<B>,
    /// Attention dropout
    attn_dropout: Dropout,
    /// Residual dropout  
    resid_dropout: Dropout,
    /// Number of attention heads
    n_heads: usize,
    /// Embedding dimension
    d_model: usize,
    /// Head dimension
    head_dim: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, dropout: f64, device: &B::Device) -> Self {
        assert!(
            d_model % n_heads == 0,
            "d_model must be divisible by n_heads"
        );
        
        let head_dim = d_model / n_heads;
        
        // Combined QKV projection (3 * d_model for q, k, v)
        let c_attn = LinearConfig::new(d_model, 3 * d_model).init(device);
        
        // Output projection
        let c_proj = LinearConfig::new(d_model, d_model).init(device);
        
        let attn_dropout = DropoutConfig::new(dropout).init();
        let resid_dropout = DropoutConfig::new(dropout).init();
        
        Self {
            c_attn,
            c_proj,
            attn_dropout,
            resid_dropout,
            n_heads,
            d_model,
            head_dim,
        }
    }
    
    /// Forward pass with causal masking
    /// Input shape: [batch, seq_len, d_model]
    /// Output shape: [batch, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let device = x.device();
        
        // Project to Q, K, V
        let qkv = self.c_attn.forward(x);
        
        // Split into Q, K, V
        let qkv_chunks: Vec<Tensor<B, 3>> = qkv.chunk(3, 2);
        let q = qkv_chunks[0].clone();
        let k = qkv_chunks[1].clone();
        let v = qkv_chunks[2].clone();
        
        // Reshape for multi-head attention: [batch, seq_len, n_heads, head_dim]
        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        
        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        
        // Compute attention scores: [batch, n_heads, seq_len, seq_len]
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.transpose()) / scale;
        
        // Create causal mask (lower triangular)
        let mask = Self::create_causal_mask(seq_len, &device);
        
        // Apply causal mask (set future positions to -inf)
        let neg_inf = Tensor::<B, 4>::zeros([batch, self.n_heads, seq_len, seq_len], &device)
            .add_scalar(-1e9);
        // Expand mask from [seq_len, seq_len] to [batch, n_heads, seq_len, seq_len]
        let mask_4d: Tensor<B, 4, burn::tensor::Bool> = mask.unsqueeze::<3>().unsqueeze::<4>();
        let attn_weights = attn_weights.mask_where(mask_4d, neg_inf);
        
        // Softmax and dropout
        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_weights = self.attn_dropout.forward(attn_weights);
        
        // Apply attention to values: [batch, n_heads, seq_len, head_dim]
        let attn_output = attn_weights.matmul(v);
        
        // Transpose back: [batch, seq_len, n_heads, head_dim]
        let attn_output = attn_output.swap_dims(1, 2);
        
        // Reshape to [batch, seq_len, d_model]
        let attn_output = attn_output.reshape([batch, seq_len, self.d_model]);
        
        // Output projection with residual dropout
        let output = self.c_proj.forward(attn_output);
        self.resid_dropout.forward(output)
    }
    
    /// Create a causal mask where True means "should be masked" (future positions)
    fn create_causal_mask(seq_len: usize, device: &B::Device) -> Tensor<B, 2, burn::tensor::Bool> {
        // Create indices tensors for comparison
        let rows: Tensor<B, 2, burn::tensor::Int> = Tensor::arange(0..seq_len as i64, device)
            .reshape([seq_len, 1])
            .repeat_dim(1, seq_len);
        let cols: Tensor<B, 2, burn::tensor::Int> = Tensor::arange(0..seq_len as i64, device)
            .reshape([1, seq_len])
            .repeat_dim(0, seq_len);
        
        // Mask where col > row (future positions)
        cols.greater(rows)
    }
}

