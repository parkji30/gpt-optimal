//! Custom Flash Attention implementation.
//!
//! This implements the Flash Attention algorithm for memory-efficient attention computation.
//! Based on the paper "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
//!
//! Key optimizations:
//! - Tiled computation to reduce memory footprint (avoids materializing N×N attention matrix)
//! - Online softmax computation with running max for numerical stability
//! - Causal masking for autoregressive models

use burn::prelude::*;
use burn::tensor::activation::softmax;

/// Block size for tiled Flash Attention computation.
/// Smaller blocks = less memory, larger blocks = better GPU utilization.
const FLASH_ATTENTION_BLOCK_SIZE: usize = 64;

/// Flash Attention module for memory-efficient scaled dot-product attention.
///
/// This implementation uses tiled computation to avoid materializing the full
/// N×N attention matrix, reducing memory complexity from O(N²) to O(N).
#[derive(Module, Debug)]
pub struct FlashAttention<B: Backend> {
    /// Number of attention heads
    n_head: usize,
    /// Dimension per head
    head_dim: usize,
    /// Dropout rate (applied during training)
    dropout: f64,
    /// Scaling factor: 1/sqrt(head_dim)
    scale: f32,
    /// Phantom marker for backend
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> FlashAttention<B> {
    /// Create a new Flash Attention module.
    pub fn new(n_head: usize, head_dim: usize, dropout: f64) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            n_head,
            head_dim,
            dropout,
            scale,
            _backend: std::marker::PhantomData,
        }
    }

    /// Compute causal self-attention using the Flash Attention algorithm.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, n_head, seq_len, head_dim]
    /// * `k` - Key tensor [batch, n_head, seq_len, head_dim]
    /// * `v` - Value tensor [batch, n_head, seq_len, head_dim]
    /// * `is_causal` - Whether to apply causal masking
    ///
    /// # Returns
    /// * Attention output [batch, n_head, seq_len, head_dim]
    pub fn forward(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        is_causal: bool,
    ) -> Tensor<B, 4> {
        let [batch_size, n_head, seq_len, head_dim] = q.dims();
        let device = q.device();

        // For small sequences, use standard attention (more efficient)
        if seq_len <= FLASH_ATTENTION_BLOCK_SIZE * 2 {
            return self.standard_attention(q, k, v, is_causal);
        }

        // Flash Attention: tiled computation
        self.tiled_flash_attention(q, k, v, batch_size, n_head, seq_len, head_dim, is_causal, device)
    }

    /// Standard attention computation for small sequences.
    /// This is used when sequence length is small enough that the full attention
    /// matrix fits comfortably in memory.
    fn standard_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        is_causal: bool,
    ) -> Tensor<B, 4> {
        let [_batch_size, _n_head, seq_len, _head_dim] = q.dims();
        let device = q.device();

        // Compute attention scores: Q @ K^T * scale
        // [batch, n_head, seq_len, head_dim] @ [batch, n_head, head_dim, seq_len]
        // -> [batch, n_head, seq_len, seq_len]
        let k_t = k.swap_dims(2, 3);
        let mut scores = q.matmul(k_t).mul_scalar(self.scale);

        // Apply causal mask if needed
        if is_causal {
            scores = self.apply_causal_mask(scores, seq_len, &device);
        }

        // Softmax along last dimension
        let attn_weights = softmax(scores, 3);

        // Apply attention to values
        // [batch, n_head, seq_len, seq_len] @ [batch, n_head, seq_len, head_dim]
        // -> [batch, n_head, seq_len, head_dim]
        attn_weights.matmul(v)
    }

    /// Tiled Flash Attention for memory-efficient computation on long sequences.
    ///
    /// This implements the core Flash Attention algorithm:
    /// 1. Process query blocks sequentially
    /// 2. For each query block, iterate over key/value blocks
    /// 3. Maintain running statistics (max, sum) for online softmax
    /// 4. Update output incrementally
    fn tiled_flash_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        batch_size: usize,
        n_head: usize,
        seq_len: usize,
        head_dim: usize,
        is_causal: bool,
        device: B::Device,
    ) -> Tensor<B, 4> {
        let block_size = FLASH_ATTENTION_BLOCK_SIZE;
        let num_blocks = (seq_len + block_size - 1) / block_size;

        // Initialize output tensor
        let mut output = Tensor::<B, 4>::zeros([batch_size, n_head, seq_len, head_dim], &device);

        // Process query blocks
        for q_block_idx in 0..num_blocks {
            let q_start = q_block_idx * block_size;
            let q_end = std::cmp::min((q_block_idx + 1) * block_size, seq_len);
            let q_len = q_end - q_start;

            // Extract query block: [batch, n_head, q_len, head_dim]
            let q_block = q.clone().slice([
                0..batch_size,
                0..n_head,
                q_start..q_end,
                0..head_dim,
            ]);

            // Initialize running statistics for online softmax
            let mut max_scores = Tensor::<B, 3>::full(
                [batch_size, n_head, q_len],
                f32::NEG_INFINITY,
                &device,
            );
            let mut sum_exp = Tensor::<B, 3>::zeros([batch_size, n_head, q_len], &device);
            let mut output_block = Tensor::<B, 4>::zeros([batch_size, n_head, q_len, head_dim], &device);

            // Determine the range of key/value blocks to process
            let kv_block_end = if is_causal { q_block_idx + 1 } else { num_blocks };

            for kv_block_idx in 0..kv_block_end {
                let kv_start = kv_block_idx * block_size;
                let kv_end = std::cmp::min((kv_block_idx + 1) * block_size, seq_len);
                let kv_len = kv_end - kv_start;

                // Extract key and value blocks
                let k_block = k.clone().slice([
                    0..batch_size,
                    0..n_head,
                    kv_start..kv_end,
                    0..head_dim,
                ]);
                let v_block = v.clone().slice([
                    0..batch_size,
                    0..n_head,
                    kv_start..kv_end,
                    0..head_dim,
                ]);

                // Compute attention scores for this block pair
                let k_block_t = k_block.swap_dims(2, 3);
                let mut block_scores = q_block.clone().matmul(k_block_t).mul_scalar(self.scale);

                // Apply causal mask within the block if needed
                if is_causal && q_block_idx == kv_block_idx {
                    block_scores = self.apply_block_causal_mask(
                        block_scores, q_len, kv_len, q_start, kv_start, &device
                    );
                }

                // Online softmax update (Flash Attention algorithm)
                // block_max: [batch, n_head, q_len, 1] -> squeeze to [batch, n_head, q_len]
                let block_max_4d = block_scores.clone().max_dim(3);
                let block_max: Tensor<B, 3> = block_max_4d.squeeze();

                // Compute exp(scores - block_max)
                let block_max_expanded = block_max.clone().unsqueeze_dim::<4>(3).expand([batch_size, n_head, q_len, kv_len]);
                let exp_scores = (block_scores - block_max_expanded).exp();

                // Sum of exp for this block: [batch, n_head, q_len, 1] -> squeeze to [batch, n_head, q_len]
                let block_sum_exp_4d = exp_scores.clone().sum_dim(3);
                let block_sum_exp: Tensor<B, 3> = block_sum_exp_4d.squeeze();

                // Update running statistics with proper rescaling
                // max_pair on same dimension tensors (both are 3D now)
                let new_max = max_scores.clone().max_pair(block_max.clone());
                let old_scale = (max_scores.clone() - new_max.clone()).exp();
                let new_scale = (block_max - new_max.clone()).exp();

                let new_sum_exp = sum_exp.clone() * old_scale.clone() + block_sum_exp * new_scale.clone();

                // Update output with rescaling - expand 3D tensors to 4D
                let old_scale_4d = old_scale.unsqueeze_dim::<4>(3).expand([batch_size, n_head, q_len, head_dim]);
                let new_scale_4d = new_scale.unsqueeze_dim::<4>(3).expand([batch_size, n_head, q_len, head_dim]);

                let weighted_v = exp_scores.matmul(v_block);
                output_block = output_block * old_scale_4d + weighted_v * new_scale_4d;

                max_scores = new_max;
                sum_exp = new_sum_exp;
            }

            // Normalize by sum_exp
            let sum_exp_4d = sum_exp.unsqueeze_dim::<4>(3).expand([batch_size, n_head, q_len, head_dim]);
            let normalized_output = output_block / sum_exp_4d;

            // Write output block back
            output = self.write_output_block(
                output, normalized_output, batch_size, n_head, seq_len, head_dim, q_start, q_end, &device
            );
        }

        output
    }

    /// Apply causal mask to attention scores.
    fn apply_causal_mask(
        &self,
        scores: Tensor<B, 4>,
        seq_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [batch_size, n_head, _, _] = scores.dims();
        
        // Create causal mask
        let mask = self.create_causal_mask(seq_len, device);
        let mask_4d: Tensor<B, 4, Bool> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0)
            .expand([batch_size, n_head, seq_len, seq_len]);
        
        let neg_inf = Tensor::<B, 4>::full(scores.dims(), f32::NEG_INFINITY, device);
        scores.mask_where(mask_4d, neg_inf)
    }

    /// Apply causal mask within a block for tiled attention.
    fn apply_block_causal_mask(
        &self,
        scores: Tensor<B, 4>,
        q_len: usize,
        kv_len: usize,
        q_start: usize,
        kv_start: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [batch_size, n_head, _, _] = scores.dims();
        
        // Create mask for this block
        let mut mask_data = vec![0i64; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                let q_pos = q_start + i;
                let k_pos = kv_start + j;
                mask_data[i * kv_len + j] = if q_pos < k_pos { 1 } else { 0 };
            }
        }
        
        let mask_flat = Tensor::<B, 1, Int>::from_data(mask_data.as_slice(), device);
        let mask_2d = mask_flat.reshape([q_len, kv_len]);
        let mask_bool = mask_2d.greater_elem(0);
        let mask_4d: Tensor<B, 4, Bool> = mask_bool.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0)
            .expand([batch_size, n_head, q_len, kv_len]);
        
        let neg_inf = Tensor::<B, 4>::full(scores.dims(), f32::NEG_INFINITY, device);
        scores.mask_where(mask_4d, neg_inf)
    }

    /// Create causal mask tensor.
    fn create_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
        let mut mask_data = vec![0i64; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                mask_data[i * seq_len + j] = if j > i { 1 } else { 0 };
            }
        }
        
        let mask_flat = Tensor::<B, 1, Int>::from_data(mask_data.as_slice(), device);
        let mask_2d = mask_flat.reshape([seq_len, seq_len]);
        mask_2d.greater_elem(0)
    }

    /// Write output block back to the full output tensor.
    fn write_output_block(
        &self,
        output: Tensor<B, 4>,
        block: Tensor<B, 4>,
        batch_size: usize,
        n_head: usize,
        seq_len: usize,
        head_dim: usize,
        start: usize,
        end: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        // Pad block to full size
        let zeros_before = Tensor::<B, 4>::zeros([batch_size, n_head, start, head_dim], device);
        let zeros_after = Tensor::<B, 4>::zeros([batch_size, n_head, seq_len - end, head_dim], device);
        
        let padded_block = if start > 0 && end < seq_len {
            Tensor::cat(vec![zeros_before, block, zeros_after], 2)
        } else if start > 0 {
            Tensor::cat(vec![zeros_before, block], 2)
        } else if end < seq_len {
            Tensor::cat(vec![block, zeros_after], 2)
        } else {
            block
        };
        
        // Create mask for the positions we're updating
        let mut mask_data = vec![0i64; seq_len];
        for i in start..end {
            mask_data[i] = 1;
        }
        let mask_1d = Tensor::<B, 1, Int>::from_data(mask_data.as_slice(), device);
        let mask_bool = mask_1d.greater_elem(0);
        let mask_4d: Tensor<B, 4, Bool> = mask_bool
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(3)
            .expand([batch_size, n_head, seq_len, head_dim]);
        
        output.mask_where(mask_4d, padded_block)
    }
}

/// Causal Self-Attention module using Flash Attention.
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    /// Combined QKV projection
    c_attn: burn::nn::Linear<B>,
    /// Output projection
    c_proj: burn::nn::Linear<B>,
    /// Flash Attention core
    flash_attn: FlashAttention<B>,
    /// Number of heads
    n_head: usize,
    /// Embedding dimension
    n_embd: usize,
    /// Head dimension
    head_dim: usize,
    /// Residual dropout
    resid_dropout: burn::nn::Dropout,
}

/// Configuration for CausalSelfAttention
#[derive(Config, Debug)]
pub struct CausalSelfAttentionConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub dropout: f64,
}

impl CausalSelfAttentionConfig {
    /// Initialize the CausalSelfAttention module
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        assert!(
            self.n_embd % self.n_head == 0,
            "n_embd must be divisible by n_head"
        );

        let head_dim = self.n_embd / self.n_head;

        let c_attn = burn::nn::LinearConfig::new(self.n_embd, 3 * self.n_embd)
            .with_bias(true)
            .init(device);

        let c_proj = burn::nn::LinearConfig::new(self.n_embd, self.n_embd)
            .with_bias(true)
            .init(device);

        let flash_attn = FlashAttention::new(self.n_head, head_dim, self.dropout);
        let resid_dropout = burn::nn::DropoutConfig::new(self.dropout).init();

        CausalSelfAttention {
            c_attn,
            c_proj,
            flash_attn,
            n_head: self.n_head,
            n_embd: self.n_embd,
            head_dim,
            resid_dropout,
        }
    }
}

impl<B: Backend> CausalSelfAttention<B> {
    /// Forward pass through causal self-attention.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _n_embd] = x.dims();

        // Project to Q, K, V
        let qkv = self.c_attn.forward(x);

        // Split into Q, K, V manually (burn 0.20 compatible)
        let q = qkv.clone().slice([0..batch_size, 0..seq_len, 0..self.n_embd]);
        let k = qkv.clone().slice([0..batch_size, 0..seq_len, self.n_embd..(2*self.n_embd)]);
        let v = qkv.slice([0..batch_size, 0..seq_len, (2*self.n_embd)..(3*self.n_embd)]);

        // Reshape for multi-head attention
        let q = q.reshape([batch_size, seq_len, self.n_head, self.head_dim]).swap_dims(1, 2);
        let k = k.reshape([batch_size, seq_len, self.n_head, self.head_dim]).swap_dims(1, 2);
        let v = v.reshape([batch_size, seq_len, self.n_head, self.head_dim]).swap_dims(1, 2);

        // Apply Flash Attention (causal)
        let y = self.flash_attn.forward(q, k, v, true);

        // Transpose back and reshape
        let y = y.swap_dims(1, 2).reshape([batch_size, seq_len, self.n_embd]);

        // Output projection and dropout
        let y = self.c_proj.forward(y);
        self.resid_dropout.forward(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_flash_attention_basic() {
        let device = Default::default();
        let flash_attn: FlashAttention<TestBackend> = FlashAttention::new(4, 32, 0.0);

        let q = Tensor::random([2, 4, 16, 32], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let k = Tensor::random([2, 4, 16, 32], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let v = Tensor::random([2, 4, 16, 32], burn::tensor::Distribution::Normal(0.0, 1.0), &device);

        let output = flash_attn.forward(q, k, v, true);

        assert_eq!(output.dims(), [2, 4, 16, 32]);
    }

    #[test]
    fn test_causal_self_attention() {
        let device = Default::default();
        let config = CausalSelfAttentionConfig {
            n_embd: 128,
            n_head: 4,
            dropout: 0.0,
        };
        let attn: CausalSelfAttention<TestBackend> = config.init(&device);

        let x = Tensor::random([2, 16, 128], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = attn.forward(x);

        assert_eq!(output.dims(), [2, 16, 128]);
    }
}
