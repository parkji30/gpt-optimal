use burn::{
    config::Config,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    tensor::{backend::{Backend, AutodiffBackend}, Int, Tensor},
    train::{
        ClassificationOutput, TrainOutput, TrainStep, ValidStep,
    },
};

use crate::model::GPT2;
use crate::data::TextBatch;

/// Configuration for the training module
#[derive(Config)]
pub struct GPT2TrainingModuleConfig {
    /// Vocabulary size for loss ignore index
    #[config(default = 50257)]
    pub pad_token_id: usize,
}

/// Training module wrapper for GPT-2
#[derive(Module, Debug)]
pub struct GPT2TrainingModule<B: Backend> {
    pub model: GPT2<B>,
    pad_token_id: usize,
}

impl<B: Backend> GPT2TrainingModule<B> {
    pub fn new(model: GPT2<B>, pad_token_id: usize) -> Self {
        Self { model, pad_token_id }
    }
    
    /// Forward pass with loss calculation
    pub fn forward(&self, batch: TextBatch<B>) -> ClassificationOutput<B> {
        let logits = self.model.forward(batch.input_ids.clone());
        let [batch_size, seq_len, vocab_size] = logits.dims();
        
        // Reshape for cross-entropy
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.target_ids.clone().reshape([batch_size * seq_len]);
        
        // Calculate loss (CrossEntropyLoss ignores pad tokens automatically if configured)
        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token_id]))
            .init(&logits_flat.device())
            .forward(logits_flat.clone(), targets_flat.clone());
        
        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }
}

impl<B: AutodiffBackend> TrainStep<TextBatch<B>, ClassificationOutput<B>> for GPT2TrainingModule<B> {
    fn step(&self, batch: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward(batch);
        let grads = output.loss.backward();
        
        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> ValidStep<TextBatch<B>, ClassificationOutput<B>> for GPT2TrainingModule<B> {
    fn step(&self, batch: TextBatch<B>) -> ClassificationOutput<B> {
        self.forward(batch)
    }
}

/// Calculate perplexity from loss
pub fn perplexity(loss: f64) -> f64 {
    loss.exp()
}

/// Learning rate scheduler with warmup
pub struct WarmupCosineScheduler {
    warmup_steps: usize,
    total_steps: usize,
    base_lr: f64,
    min_lr: f64,
    current_step: usize,
}

impl WarmupCosineScheduler {
    pub fn new(warmup_steps: usize, total_steps: usize, base_lr: f64, min_lr: f64) -> Self {
        Self {
            warmup_steps,
            total_steps,
            base_lr,
            min_lr,
            current_step: 0,
        }
    }
    
    pub fn step(&mut self) -> f64 {
        let lr = if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            // Cosine decay
            let progress = (self.current_step - self.warmup_steps) as f64
                / (self.total_steps - self.warmup_steps) as f64;
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
        };
        
        self.current_step += 1;
        lr
    }
    
    pub fn current_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            let progress = (self.current_step - self.warmup_steps) as f64
                / (self.total_steps - self.warmup_steps) as f64;
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }
}


