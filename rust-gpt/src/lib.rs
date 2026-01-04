//! # Rust GPT-2
//! 
//! A minimal GPT-2 implementation using the Burn deep learning framework.
//! 
//! ## Features
//! 
//! - GPT-2 model architecture (configurable sizes: tiny, small, medium)
//! - Training pipeline with Burn's learner
//! - Text generation with temperature sampling, top-k, and top-p
//! - Real-time training visualization
//! - Dataset loading and preprocessing
//! 
//! ## Usage
//! 
//! ```rust,ignore
//! use rust_gpt::{
//!     model::{GPT2, GPT2Config},
//!     training::TrainingConfig,
//!     tokenizer::Tokenizer,
//! };
//! 
//! // Create a small GPT-2 model
//! let config = GPT2Config::small();
//! let device = Default::default();
//! let model = GPT2::new(&config, &device);
//! ```

#![recursion_limit = "256"]

pub mod model;
pub mod tokenizer;
pub mod data;
pub mod training;
pub mod eval;
pub mod visualization;

// Re-exports for convenience
pub use model::{GPT2, GPT2Config};
pub use tokenizer::Tokenizer;
pub use data::{TextDataset, TextDatasetConfig, TextBatcher, TextBatch};
pub use training::{TrainingConfig, GPT2TrainingModule};
pub use eval::{TextGenerator, EvalMetrics};
pub use visualization::TrainingDashboard;

