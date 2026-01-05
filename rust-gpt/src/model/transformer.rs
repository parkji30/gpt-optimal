use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use super::attention::CausalSelfAttention;

/// Feed-Forward Network (MLP) block
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    c_fc: Linear<B>,
    c_proj: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> MLP<B> {
    pub fn new(d_model: usize, d_ff: usize, dropout: f64, device: &B::Device) -> Self {
        let c_fc = LinearConfig::new(d_model, d_ff).init(device);
        let c_proj = LinearConfig::new(d_ff, d_model).init(device);
        let dropout = DropoutConfig::new(dropout).init();
        
        Self { c_fc, c_proj, dropout }
    }
    
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        // GELU activation (GPT-2 uses approximate GELU)
        let x = burn::tensor::activation::gelu(x);
        let x = self.c_proj.forward(x);
        self.dropout.forward(x)
    }
}

/// Transformer Block (Pre-LN variant as in GPT-2)
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    ln_1: LayerNorm<B>,
    attn: CausalSelfAttention<B>,
    ln_2: LayerNorm<B>,
    mlp: MLP<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        layer_norm_eps: f64,
        device: &B::Device,
    ) -> Self {
        let ln_1 = LayerNormConfig::new(d_model)
            .with_epsilon(layer_norm_eps)
            .init(device);
        let attn = CausalSelfAttention::new(d_model, n_heads, dropout, device);
        let ln_2 = LayerNormConfig::new(d_model)
            .with_epsilon(layer_norm_eps)
            .init(device);
        let mlp = MLP::new(d_model, d_ff, dropout, device);
        
        Self { ln_1, attn, ln_2, mlp }
    }
    
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-LN with residual connections
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x.clone()));
        x.clone() + self.mlp.forward(self.ln_2.forward(x))
    }
}


