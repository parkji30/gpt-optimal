use crate::model::GPT2Config;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Model configuration
    pub model: GPT2Config,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Number of epochs
    pub num_epochs: usize,
    
    /// Number of workers for data loading
    pub num_workers: usize,
    
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    
    /// Weight decay
    pub weight_decay: f64,
    
    /// Warmup steps
    pub warmup_steps: usize,
    
    /// Random seed
    pub seed: u64,
    
    /// Checkpoint directory
    pub checkpoint_dir: String,
    
    /// Log interval (steps)
    pub log_interval: usize,
    
    /// Save interval (steps)
    pub save_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: GPT2Config::tiny(),
            learning_rate: 3e-4,
            batch_size: 32,
            num_epochs: 10,
            num_workers: 4,
            gradient_accumulation_steps: 1,
            weight_decay: 0.01,
            warmup_steps: 100,
            seed: 42,
            checkpoint_dir: "checkpoints".to_string(),
            log_interval: 10,
            save_interval: 500,
        }
    }
}

impl TrainingConfig {
    /// Create a default training config with small model
    pub fn small() -> Self {
        Self {
            model: GPT2Config::small(),
            learning_rate: 3e-4,
            batch_size: 16,
            num_epochs: 10,
            num_workers: 4,
            gradient_accumulation_steps: 1,
            weight_decay: 0.01,
            warmup_steps: 100,
            seed: 42,
            checkpoint_dir: "checkpoints".to_string(),
            log_interval: 10,
            save_interval: 500,
        }
    }
    
    /// Create a tiny training config for quick testing
    pub fn tiny() -> Self {
        Self {
            model: GPT2Config::tiny(),
            learning_rate: 1e-3,
            batch_size: 8,
            num_epochs: 5,
            num_workers: 2,
            gradient_accumulation_steps: 1,
            weight_decay: 0.01,
            warmup_steps: 50,
            seed: 42,
            checkpoint_dir: "checkpoints".to_string(),
            log_interval: 5,
            save_interval: 100,
        }
    }
}

