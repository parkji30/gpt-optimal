//! Text generation binary for GPT-2
//! 
//! Usage:
//!   cargo run --bin generate -- --prompt "Hello world" --max-tokens 50

use std::path::PathBuf;

use burn::backend::Wgpu;
use clap::Parser;

use rust_gpt::{
    model::{GPT2, GPT2Config},
    tokenizer::Tokenizer,
    eval::TextGenerator,
};

type Backend = Wgpu;

#[derive(Parser)]
#[command(name = "generate")]
#[command(about = "Generate text using a GPT-2 model")]
struct Args {
    /// Text prompt to continue
    #[arg(short, long, default_value = "The")]
    prompt: String,
    
    /// Maximum number of tokens to generate
    #[arg(short, long, default_value = "50")]
    max_tokens: usize,
    
    /// Sampling temperature (higher = more random)
    #[arg(short, long, default_value = "1.0")]
    temperature: f64,
    
    /// Top-k sampling (only sample from top k tokens)
    #[arg(short = 'k', long)]
    top_k: Option<usize>,
    
    /// Top-p (nucleus) sampling (sample from tokens with cumulative prob >= p)
    #[arg(short = 'n', long)]
    top_p: Option<f64>,
    
    /// Use greedy decoding instead of sampling
    #[arg(short, long)]
    greedy: bool,
    
    /// Path to model checkpoint (optional)
    #[arg(short, long)]
    model: Option<PathBuf>,
    
    /// Model size (tiny, small) - used when no checkpoint provided
    #[arg(short = 's', long, default_value = "tiny")]
    model_size: String,
    
    /// Number of samples to generate
    #[arg(short = 'c', long, default_value = "1")]
    count: usize,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .init();
    
    let args = Args::parse();
    
    println!("╔══════════════════════════════════════════════════╗");
    println!("║         GPT-2 Text Generation                    ║");
    println!("╚══════════════════════════════════════════════════╝\n");
    
    let device = Default::default();
    let tokenizer = Tokenizer::new();
    
    // Create model config
    let config = match args.model_size.as_str() {
        "tiny" => GPT2Config::tiny(),
        "small" => GPT2Config::small(),
        _ => GPT2Config::tiny(),
    };
    
    println!("Model: {} ({} layers, {} dim)", 
        args.model_size, config.n_layers, config.d_model);
    println!("Prompt: \"{}\"", args.prompt);
    println!("Max tokens: {}", args.max_tokens);
    
    if args.greedy {
        println!("Decoding: greedy");
    } else {
        println!("Temperature: {}", args.temperature);
        if let Some(k) = args.top_k {
            println!("Top-k: {}", k);
        }
        if let Some(p) = args.top_p {
            println!("Top-p: {}", p);
        }
    }
    println!();
    
    // Create model
    let model: GPT2<Backend> = GPT2::new(&config, &device);
    let generator = TextGenerator::new(model, tokenizer, device);
    
    println!("{}", "─".repeat(50));
    
    for i in 0..args.count {
        if args.count > 1 {
            println!("\n[Sample {}]", i + 1);
        }
        
        let output = if args.greedy {
            generator.generate_greedy(&args.prompt, args.max_tokens)
        } else {
            generator.generate(
                &args.prompt,
                args.max_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
            )
        };
        
        println!("{}", output);
    }
    
    println!("{}", "─".repeat(50));
}

