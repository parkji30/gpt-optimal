//! Configuration structures mirroring the Python implementation.

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

/// GPT-2 model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block_size: usize,
    pub dropout: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            n_embd: 128,
            n_head: 4,
            n_layer: 4,
            block_size: 128,
            dropout: 0.1,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_iters: usize,
    pub eval_interval: usize,
    pub eval_iters: usize,
    pub gradient_accumulation_steps: usize,
    pub max_grad_norm: f64,
    pub warmup_iters: usize,
    pub lr_decay_iters: usize,
    pub min_lr: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 3e-4,
            max_iters: 1000,
            eval_interval: 200,
            eval_iters: 100,
            gradient_accumulation_steps: 1,
            max_grad_norm: 1.0,
            warmup_iters: 100,
            lr_decay_iters: 1000,
            min_lr: 3e-5,
        }
    }
}

/// Data configuration
#[derive(Debug, Clone, Deserialize)]
pub struct DataConfig {
    pub train_split: f64,
    pub path: String,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train_split: 0.9,
            path: "../data/shakespeare.txt".to_string(),
        }
    }
}

/// Generation configuration
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 500,
            temperature: 0.8,
        }
    }
}

/// AMP (Automatic Mixed Precision) configuration
#[derive(Debug, Clone, Deserialize)]
pub struct AmpConfig {
    pub enabled: bool,
    pub init_scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: usize,
}

impl Default for AmpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }
}

/// Optimization configuration
#[derive(Debug, Clone, Deserialize)]
pub struct OptimizationConfig {
    pub compile: bool,
    pub compile_mode: String,
    pub cudnn_benchmark: bool,
    pub tf32: bool,
    pub flash_attention: bool,
    pub fused_optimizer: bool,
    pub pin_memory: bool,
    pub num_workers: usize,
    pub persistent_workers: bool,
    pub prefetch_factor: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            compile: true,
            compile_mode: "default".to_string(),
            cudnn_benchmark: true,
            tf32: true,
            flash_attention: true,
            fused_optimizer: true,
            pin_memory: true,
            num_workers: 2,
            persistent_workers: true,
            prefetch_factor: 2,
        }
    }
}

/// Complete configuration loaded from JSON
#[derive(Debug, Clone, Deserialize)]
pub struct FullConfig {
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub data: DataConfig,
    pub generation: GenerationConfig,
    #[serde(default)]
    pub amp: AmpConfig,
    #[serde(default)]
    pub optimization: OptimizationConfig,
}

impl FullConfig {
    /// Load configuration from JSON file
    pub fn from_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: FullConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
}
