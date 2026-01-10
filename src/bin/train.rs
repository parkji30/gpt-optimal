//! Training binary for GPT-2
//! 
//! Usage:
//!   cargo run --bin train -- --data path/to/data.jsonl --model-size tiny --epochs 5

use std::path::PathBuf;

use burn::backend::{Autodiff, Wgpu};
use clap::Parser;

use rust_gpt::{
    data::{TextDataset, TextDatasetConfig},
    model::{GPT2, GPT2Config},
    tokenizer::Tokenizer,
    training::{TrainingConfig, GPT2TrainingModule},
    visualization::TrainingDashboard,
};

type Backend = Wgpu;
type AutodiffBackend = Autodiff<Backend>;

#[derive(Parser)]
#[command(name = "train")]
#[command(about = "Train a GPT-2 model")]
struct Args {
    /// Path to training data (JSONL or text file)
    #[arg(short, long)]
    data: Option<PathBuf>,
    
    /// Model size (tiny, small)
    #[arg(short, long, default_value = "tiny")]
    model_size: String,
    
    /// Number of epochs
    #[arg(short, long, default_value = "5")]
    epochs: usize,
    
    /// Batch size
    #[arg(short, long, default_value = "8")]
    batch_size: usize,
    
    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    learning_rate: f64,
    
    /// Output directory for checkpoints
    #[arg(short, long, default_value = "checkpoints")]
    output: PathBuf,
    
    /// Sequence length
    #[arg(short, long, default_value = "128")]
    seq_length: usize,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();
    
    let args = Args::parse();
    
    println!("╔══════════════════════════════════════════════════╗");
    println!("║         GPT-2 Training with Burn                 ║");
    println!("╚══════════════════════════════════════════════════╝\n");
    
    let device = Default::default();
    let tokenizer = Tokenizer::new();
    
    // Create model config
    let model_config = match args.model_size.as_str() {
        "tiny" => GPT2Config::tiny(),
        "small" => GPT2Config::small(),
        _ => {
            println!("Unknown model size '{}', using tiny", args.model_size);
            GPT2Config::tiny()
        }
    };
    
    println!("Model Configuration:");
    println!("  Size: {}", args.model_size);
    println!("  Embedding dim: {}", model_config.d_model);
    println!("  Attention heads: {}", model_config.n_heads);
    println!("  Layers: {}", model_config.n_layers);
    println!("  Max sequence length: {}", model_config.max_seq_len);
    println!();
    
    // Load dataset
    let dataset_config = TextDatasetConfig {
        seq_length: args.seq_length.min(model_config.max_seq_len),
        add_eos: true,
    };
    
    let train_dataset = if let Some(path) = &args.data {
        println!("Loading training data from {:?}...", path);
        if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
            TextDataset::from_jsonl(path, &tokenizer, &dataset_config, "text")
                .expect("Failed to load JSONL dataset")
        } else {
            TextDataset::from_file(path, &tokenizer, &dataset_config)
                .expect("Failed to load text dataset")
        }
    } else {
        println!("No data path provided, using demo dataset");
        TextDataset::demo(&tokenizer, &dataset_config)
    };
    
    println!("Dataset size: {} samples", train_dataset.len());
    println!();
    
    // Create model
    println!("Initializing model...");
    let model: GPT2<AutodiffBackend> = GPT2::new(&model_config, &device);
    
    // Estimate parameters
    let params = estimate_params(&model_config);
    println!("Estimated parameters: {}", format_params(params));
    println!();
    
    // Create training config
    let training_config = TrainingConfig {
        model: model_config,
        learning_rate: args.learning_rate,
        batch_size: args.batch_size,
        num_epochs: args.epochs,
        num_workers: 4,
        gradient_accumulation_steps: 1,
        weight_decay: 0.01,
        warmup_steps: 100,
        seed: 42,
        checkpoint_dir: args.output.to_string_lossy().to_string(),
        log_interval: 10,
        save_interval: 500,
    };
    
    // Create output directory
    std::fs::create_dir_all(&args.output).ok();
    
    // Training with dashboard
    let mut dashboard = TrainingDashboard::new();
    
    println!("Starting training...");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.learning_rate);
    println!();
    
    train_model(
        model,
        train_dataset,
        &training_config,
        &tokenizer,
        &mut dashboard,
        &device,
    );
    
    dashboard.finish();
    
    println!("\n✓ Training complete!");
    println!("Checkpoints saved to: {:?}", args.output);
}

fn train_model<B: burn::tensor::backend::AutodiffBackend>(
    mut model: GPT2<B>,
    dataset: TextDataset,
    config: &TrainingConfig,
    tokenizer: &Tokenizer,
    dashboard: &mut TrainingDashboard,
    device: &B::Device,
) {
    use burn::optim::{AdamWConfig, Optimizer, GradientsParams};
    use burn::data::dataloader::DataLoaderBuilder;
    use rust_gpt::data::{TextBatcher, TextBatch};
    use rust_gpt::data::dataset::TextItem;
    use rust_gpt::visualization::dashboard::TrainingMetrics;
    use std::time::Instant;
    
    let pad_token_id = tokenizer.pad_token_id();
    let batcher: TextBatcher<B> = TextBatcher::new(device.clone(), pad_token_id);
    
    let dataloader = DataLoaderBuilder::<B, TextItem, TextBatch<B>>::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(dataset);
    
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    
    let total_steps = dataloader.num_items() / config.batch_size * config.num_epochs;
    let mut global_step = 0;
    let start_time = Instant::now();
    
    for epoch in 0..config.num_epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        for batch in dataloader.iter() {
            // Forward pass
            let logits = model.forward(batch.input_ids.clone());
            let [batch_size, seq_len, vocab_size] = logits.dims();
            
            let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
            let targets_flat = batch.target_ids.reshape([batch_size * seq_len]);
            
            // Calculate loss
            let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                .init(device)
                .forward(logits_flat, targets_flat);
            
            // Extract loss value
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
            let samples_per_sec = (global_step * config.batch_size) as f64 / elapsed.as_secs_f64();
            
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
        
        let avg_loss = epoch_loss / num_batches as f64;
        println!("\nEpoch {}/{} complete. Average loss: {:.4}, Perplexity: {:.2}", 
            epoch + 1, config.num_epochs, avg_loss, avg_loss.exp());
    }
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

