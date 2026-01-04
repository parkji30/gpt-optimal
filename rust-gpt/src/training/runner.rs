//! Simple training runner for GPT-2
//! 
//! This provides a straightforward training loop without the full Burn learner infrastructure.

use std::path::Path;
use std::time::Instant;

use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};

use crate::{
    data::{TextBatcher, TextBatch, TextDataset, TextDatasetConfig},
    data::dataset::TextItem,
    model::{GPT2, GPT2Config},
    tokenizer::Tokenizer,
    training::TrainingConfig,
    visualization::dashboard::{TrainingDashboard, TrainingMetrics},
};

/// Simple training loop
pub fn train_simple<B: AutodiffBackend>(
    mut model: GPT2<B>,
    train_dataset: TextDataset,
    config: &TrainingConfig,
    device: &B::Device,
) -> GPT2<B> {
    let tokenizer = Tokenizer::new();
    let pad_token_id = tokenizer.pad_token_id();
    
    let batcher: TextBatcher<B> = TextBatcher::new(device.clone(), pad_token_id);
    
    let dataloader = DataLoaderBuilder::<B, TextItem, TextBatch<B>>::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);
    
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    
    let total_batches = dataloader.num_items() / config.batch_size;
    let total_steps = total_batches * config.num_epochs;
    let mut global_step = 0;
    let start_time = Instant::now();
    
    let mut dashboard = TrainingDashboard::new();
    
    println!("Training configuration:");
    println!("  Epochs: {}", config.num_epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Total steps: {}", total_steps);
    println!();
    
    for epoch in 0..config.num_epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        for batch in dataloader.iter() {
            // Forward pass
            let logits = model.forward(batch.input_ids.clone());
            let [batch_size, seq_len, vocab_size] = logits.dims();
            
            // Reshape for loss calculation
            let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
            let targets_flat = batch.target_ids.reshape([batch_size * seq_len]);
            
            // Calculate cross-entropy loss
            let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                .with_pad_tokens(Some(vec![pad_token_id]))
                .init(device)
                .forward(logits_flat, targets_flat);
            
            // Extract scalar loss value
            let loss_data = loss.clone().into_data();
            let loss_value: f32 = loss_data.to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_value as f64;
            num_batches += 1;
            
            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            
            // Update weights
            model = optimizer.step(config.learning_rate, model, grads);
            
            global_step += 1;
            
            // Update dashboard
            let elapsed = start_time.elapsed();
            let samples_per_sec = (global_step * config.batch_size) as f64 / elapsed.as_secs_f64().max(0.001);
            
            let metrics = TrainingMetrics {
                epoch,
                step: global_step,
                total_steps,
                train_loss: loss_value as f64,
                valid_loss: None,
                learning_rate: config.learning_rate,
                samples_per_second: samples_per_sec,
                elapsed_time: elapsed,
                perplexity: (loss_value as f64).exp(),
            };
            
            dashboard.update(metrics);
        }
        
        let avg_loss = epoch_loss / num_batches.max(1) as f64;
        println!("\nEpoch {}/{} complete. Average loss: {:.4}, Perplexity: {:.2}", 
            epoch + 1, config.num_epochs, avg_loss, avg_loss.exp());
    }
    
    dashboard.finish();
    model
}

/// Save model checkpoint using Burn's record system
pub fn save_model<B: burn::tensor::backend::Backend>(
    model: &GPT2<B>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> 
where
    GPT2<B>: burn::module::Module<B>,
{
    use burn::module::Module;
    let recorder = CompactRecorder::new();
    model.clone().save_file(path, &recorder)?;
    println!("Model saved to {:?}", path);
    Ok(())
}

/// Load model checkpoint
pub fn load_model<B: burn::tensor::backend::Backend>(
    path: &Path,
    config: &GPT2Config,
    device: &B::Device,
) -> Result<GPT2<B>, Box<dyn std::error::Error>> 
where
    GPT2<B>: burn::module::Module<B>,
{
    use burn::module::Module;
    let model = GPT2::new(config, device);
    let recorder = CompactRecorder::new();
    let loaded = model.load_file(path, &recorder, device)?;
    println!("Model loaded from {:?}", path);
    Ok(loaded)
}

/// Run training with optional validation
pub fn run_training<B: AutodiffBackend>(
    config: &TrainingConfig,
    train_dataset: TextDataset,
    _valid_dataset: Option<TextDataset>,
    device: B::Device,
) -> GPT2<B> {
    println!("Creating GPT-2 model...");
    let model = GPT2::new(&config.model, &device);
    
    // Estimate parameters
    let params = estimate_params(&config.model);
    println!("Model parameters: ~{}", format_params(params));
    println!();
    
    println!("Starting training...");
    train_simple(model, train_dataset, config, &device)
}

fn estimate_params(config: &GPT2Config) -> usize {
    let token_emb = config.vocab_size * config.d_model;
    let pos_emb = config.max_seq_len * config.d_model;
    let attn = 4 * config.d_model * config.d_model * config.n_layers;
    let mlp = 2 * config.d_model * config.d_ff * config.n_layers;
    let lm_head = config.d_model * config.vocab_size;
    token_emb + pos_emb + attn + mlp + lm_head
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
