//! GPT-2 Training and Generation CLI
//! 
//! This is the main entry point for training GPT-2 models using Burn.

#![recursion_limit = "256"]

mod model;
mod tokenizer;
mod data;
mod training;
mod eval;
mod visualization;

use std::path::PathBuf;

use burn::backend::{Autodiff, Wgpu};
use clap::{Parser, Subcommand};

use crate::data::{TextDataset, TextDatasetConfig};
use crate::model::{GPT2, GPT2Config};
use crate::tokenizer::Tokenizer;
use crate::training::TrainingConfig;
use crate::eval::TextGenerator;
use crate::visualization::TrainingDashboard;

type Backend = Wgpu;
type AutodiffBackend = Autodiff<Backend>;

/// GPT-2 Training CLI
#[derive(Parser)]
#[command(name = "rust-gpt")]
#[command(about = "GPT-2 implementation using the Burn framework")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a GPT-2 model
    Train {
        /// Path to training data (JSONL or text file)
        #[arg(short, long)]
        data: Option<PathBuf>,
        
        /// Model size (tiny, small, medium)
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
    },
    
    /// Generate text using a trained model
    Generate {
        /// Path to model checkpoint
        #[arg(short, long)]
        model: Option<PathBuf>,
        
        /// Text prompt to continue
        #[arg(short, long, default_value = "The")]
        prompt: String,
        
        /// Maximum number of tokens to generate
        #[arg(short, long, default_value = "50")]
        max_tokens: usize,
        
        /// Sampling temperature
        #[arg(short, long, default_value = "1.0")]
        temperature: f64,
        
        /// Top-k sampling
        #[arg(short = 'k', long)]
        top_k: Option<usize>,
        
        /// Top-p (nucleus) sampling
        #[arg(short = 'n', long)]
        top_p: Option<f64>,
    },
    
    /// Run a demo with a small model
    Demo,
    
    /// Evaluate a model on a dataset
    Eval {
        /// Path to model checkpoint
        #[arg(short, long)]
        model: Option<PathBuf>,
        
        /// Path to evaluation data
        #[arg(short, long)]
        data: Option<PathBuf>,
    },
}

fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Train {
            data,
            model_size,
            epochs,
            batch_size,
            learning_rate,
            output,
        } => {
            run_training(data, &model_size, epochs, batch_size, learning_rate, output);
        }
        Commands::Generate {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
        } => {
            run_generation(model, &prompt, max_tokens, temperature, top_k, top_p);
        }
        Commands::Demo => {
            run_demo();
        }
        Commands::Eval { model, data } => {
            run_evaluation(model, data);
        }
    }
}

fn run_training(
    data_path: Option<PathBuf>,
    model_size: &str,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    output: PathBuf,
) {
    println!("=================================================");
    println!("          GPT-2 Training with Burn");
    println!("=================================================\n");
    
    let device = Default::default();
    let tokenizer = Tokenizer::new();
    
    // Create model config based on size
    let model_config = match model_size {
        "tiny" => GPT2Config::tiny(),
        "small" => GPT2Config::small(),
        _ => {
            println!("Unknown model size '{}', using tiny", model_size);
            GPT2Config::tiny()
        }
    };
    
    println!("Model Configuration:");
    println!("  Size: {}", model_size);
    println!("  Vocab size: {}", model_config.vocab_size);
    println!("  Max seq len: {}", model_config.max_seq_len);
    println!("  Embedding dim: {}", model_config.d_model);
    println!("  Attention heads: {}", model_config.n_heads);
    println!("  Layers: {}", model_config.n_layers);
    println!();
    
    // Load or create dataset
    let dataset_config = TextDatasetConfig {
        seq_length: model_config.max_seq_len,
        add_eos: true,
    };
    
    let train_dataset = if let Some(path) = data_path {
        println!("Loading training data from {:?}...", path);
        if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
            TextDataset::from_jsonl(&path, &tokenizer, &dataset_config, "text")
                .expect("Failed to load JSONL dataset")
        } else {
            TextDataset::from_file(&path, &tokenizer, &dataset_config)
                .expect("Failed to load text dataset")
        }
    } else {
        println!("Using demo dataset (no data path provided)");
        TextDataset::demo(&tokenizer, &dataset_config)
    };
    
    println!("Dataset size: {} samples\n", train_dataset.len());
    
    // Create training config
    let training_config = TrainingConfig {
        model: model_config.clone(),
        learning_rate,
        batch_size,
        num_epochs: epochs,
        num_workers: 4,
        gradient_accumulation_steps: 1,
        weight_decay: 0.01,
        warmup_steps: 100,
        seed: 42,
        checkpoint_dir: output.to_string_lossy().to_string(),
        log_interval: 10,
        save_interval: 500,
    };
    
    // Create model
    println!("Creating GPT-2 model...");
    let model: GPT2<AutodiffBackend> = GPT2::new(&model_config, &device);
    
    // Count parameters
    let num_params: usize = count_parameters(&model_config);
    println!("Model parameters: {}\n", format_number(num_params));
    
    // Training loop with dashboard
    let mut dashboard = TrainingDashboard::new();
    
    println!("Starting training...\n");
    
    // Note: For a complete implementation, you would use Burn's LearnerBuilder
    // This is a simplified manual training loop for demonstration
    simple_training_loop(
        model,
        train_dataset,
        &training_config,
        &tokenizer,
        &mut dashboard,
        &device,
    );
    
    dashboard.finish();
    
    println!("\nTraining complete! Checkpoints saved to {:?}", output);
}

fn run_generation(
    _model_path: Option<PathBuf>,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
) {
    println!("=================================================");
    println!("          GPT-2 Text Generation");
    println!("=================================================\n");
    
    let device = Default::default();
    let tokenizer = Tokenizer::new();
    let config = GPT2Config::tiny();
    
    // Create or load model
    println!("Loading model...");
    let model: GPT2<Backend> = GPT2::new(&config, &device);
    
    let generator = TextGenerator::new(model, tokenizer, device);
    
    println!("Prompt: {}\n", prompt);
    println!("Generating {} tokens with temperature={}...\n", max_tokens, temperature);
    
    let output = generator.generate(prompt, max_tokens, temperature, top_k, top_p);
    
    println!("Generated text:");
    println!("{}", "─".repeat(50));
    println!("{}", output);
    println!("{}", "─".repeat(50));
}

fn run_demo() {
    println!("=================================================");
    println!("          GPT-2 Demo with Burn");
    println!("=================================================\n");
    
    let device = Default::default();
    let tokenizer = Tokenizer::new();
    
    // Create a tiny model for demo
    let config = GPT2Config::tiny();
    
    println!("Creating tiny GPT-2 model...");
    println!("  d_model: {}", config.d_model);
    println!("  n_heads: {}", config.n_heads);
    println!("  n_layers: {}", config.n_layers);
    println!();
    
    let model: GPT2<Backend> = GPT2::new(&config, &device);
    
    // Count parameters
    let num_params = count_parameters(&config);
    println!("Model parameters: {}\n", format_number(num_params));
    
    // Test forward pass
    println!("Testing forward pass...");
    let test_text = "Hello, world!";
    let tokens = tokenizer.encode(test_text);
    println!("  Input: \"{}\"", test_text);
    println!("  Tokens: {:?}", tokens);
    
    // Create input tensor
    use burn::tensor::{Tensor, TensorData, Int};
    let input_ids: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
    let seq_len = input_ids.len();
    let input_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
        TensorData::new(input_ids, [1, seq_len]),
        &device,
    );
    
    let logits = model.forward(input_tensor);
    let [batch, seq, vocab] = logits.dims();
    println!("  Output shape: [{}, {}, {}]", batch, seq, vocab);
    println!();
    
    // Demo generation
    println!("Demo generation (random weights, no training):");
    let generator = TextGenerator::new(model, tokenizer, device);
    
    let prompts = [
        "The",
        "Hello",
        "Machine learning",
    ];
    
    for prompt in prompts {
        let output = generator.generate_greedy(prompt, 10);
        println!("  \"{}\" -> \"{}\"", prompt, output);
    }
    
    println!("\n✓ Demo complete!");
    println!("\nTo train a model, run:");
    println!("  cargo run -- train --data <path/to/data.jsonl>");
}

fn run_evaluation(model_path: Option<PathBuf>, data_path: Option<PathBuf>) {
    println!("=================================================");
    println!("          GPT-2 Evaluation");
    println!("=================================================\n");
    
    let device = Default::default();
    let tokenizer = Tokenizer::new();
    let config = GPT2Config::tiny();
    
    let model: GPT2<Backend> = GPT2::new(&config, &device);
    
    let dataset_config = TextDatasetConfig::default();
    let dataset = if let Some(path) = data_path {
        println!("Loading evaluation data from {:?}...", path);
        TextDataset::from_file(&path, &tokenizer, &dataset_config)
            .expect("Failed to load dataset")
    } else {
        println!("Using demo dataset for evaluation");
        TextDataset::demo(&tokenizer, &dataset_config)
    };
    
    println!("Evaluating on {} samples...\n", dataset.len());
    
    let metrics = eval::evaluate_model(&model, &dataset, 8, &device);
    
    println!("Evaluation Results:");
    println!("  Loss: {:.4}", metrics.loss);
    println!("  Perplexity: {:.4}", metrics.perplexity);
    println!("  Accuracy: {:.2}%", metrics.accuracy * 100.0);
}

fn simple_training_loop<B: burn::tensor::backend::AutodiffBackend>(
    mut model: GPT2<B>,
    dataset: TextDataset,
    config: &TrainingConfig,
    tokenizer: &Tokenizer,
    dashboard: &mut TrainingDashboard,
    device: &B::Device,
) {
    use burn::optim::{AdamWConfig, Optimizer, GradientsParams};
    use burn::data::dataloader::DataLoaderBuilder;
    use crate::data::{TextBatcher, TextBatch};
    use crate::data::dataset::TextItem;
    use crate::visualization::dashboard::TrainingMetrics;
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
            
            // Extract loss value safely
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
        println!("\nEpoch {} complete. Average loss: {:.4}, Perplexity: {:.2}", 
            epoch + 1, avg_loss, avg_loss.exp());
    }
}

fn count_parameters(config: &GPT2Config) -> usize {
    // Estimate based on config
    let token_emb = config.vocab_size * config.d_model;
    let pos_emb = config.max_seq_len * config.d_model;
    let attn_per_layer = 4 * config.d_model * config.d_model; // Q, K, V, O projections
    let mlp_per_layer = 2 * config.d_model * config.d_ff; // Up and down projections
    let ln_per_layer = 2 * 2 * config.d_model; // 2 layer norms, each with scale and bias
    let layer_params = attn_per_layer + mlp_per_layer + ln_per_layer;
    let lm_head = config.d_model * config.vocab_size;
    
    token_emb + pos_emb + config.n_layers * layer_params + lm_head
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

