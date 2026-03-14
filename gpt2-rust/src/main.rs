//! Minimal GPT-2 implementation in Rust with Burn, custom Flash Attention and AMP.
//!
//! This implementation mirrors the Python gpt2.py training loop with:
//! - Custom Flash Attention (tiled attention with memory-efficient computation)
//! - Custom Automatic Mixed Precision (gradient scaling for f16/f32 mixed precision)
//! - CUDA backend support via burn-cuda (with --features cuda)
//! - CPU backend support via burn-ndarray (with --features cpu or as fallback)

mod config;
mod model;
mod attention;
mod amp;
mod dataset;
mod training;

use anyhow::Result;
use burn::backend::Autodiff;
use std::path::PathBuf;

use crate::config::FullConfig;
use crate::training::train;

// Backend types
#[cfg(all(feature = "cuda", not(feature = "cpu")))]
use burn_cuda::{Cuda, CudaDevice};

use burn_ndarray::{NdArray, NdArrayDevice};

/// CUDA backend with autodiff support
#[cfg(all(feature = "cuda", not(feature = "cpu")))]
pub type SelectedBackend = Autodiff<Cuda>;

/// CPU backend with autodiff support
#[cfg(any(feature = "cpu", not(feature = "cuda")))]
pub type SelectedBackend = Autodiff<NdArray>;

fn main() -> Result<()> {
    // Load configuration from hyperparams/config.json (relative to project root)
    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("hyperparams/config.json");
    
    let config: FullConfig = FullConfig::from_json(&config_path)?;
    
    // Use CUDA if feature enabled and not using CPU feature
    #[cfg(all(feature = "cuda", not(feature = "cpu")))]
    {
        let device = CudaDevice::new(0);
        
        // Print device info in format matching benchmark.sh expectations
        #[cfg(feature = "amp")]
        println!("Using CUDA with AMP backend");
        #[cfg(not(feature = "amp"))]
        println!("Using CUDA backend");
        
        return train::<SelectedBackend>(&config, device);
    }
    
    // Use CPU backend
    #[cfg(any(feature = "cpu", not(feature = "cuda")))]
    {
        let device = NdArrayDevice::Cpu;
        
        // Print device info in format matching benchmark.sh expectations
        #[cfg(feature = "amp")]
        println!("Using CPU with AMP backend");
        #[cfg(not(feature = "amp"))]
        println!("Using CPU backend");
        
        train::<SelectedBackend>(&config, device)
    }
}
