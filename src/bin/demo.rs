//! Demo binary for GPT-2
//! 
//! This runs a quick demonstration of the GPT-2 implementation.

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData, Int};

use rust_gpt::{
    model::{GPT2, GPT2Config},
    tokenizer::Tokenizer,
    data::{TextDataset, TextDatasetConfig},
    eval::{TextGenerator, evaluate_model},
};

type Backend = Wgpu;

fn main() {
    println!("GPT 2 Training");
    let device = Default::default();
    let tokenizer = Tokenizer::new();
    
    // ========================================
    // 1. Model Architecture Demo
    // ========================================
    println!("1️⃣  Model Architecture\n");
    
    let config = GPT2Config::tiny();
    println!("   Creating tiny GPT-2 model:");
    println!("   ├─ Vocabulary size: {}", config.vocab_size);
    println!("   ├─ Max sequence length: {}", config.max_seq_len);
    println!("   ├─ Embedding dimension: {}", config.d_model);
    println!("   ├─ Attention heads: {}", config.n_heads);
    println!("   ├─ Transformer layers: {}", config.n_layers);
    println!("   └─ Feed-forward dimension: {}", config.d_ff);
    println!();
    
    let model: GPT2<Backend> = GPT2::new(&config, &device);
    
    // Estimate parameters
    let params = estimate_parameters(&config);
    println!("   Estimated parameters: {}\n", format_number(params));
    
    // ========================================
    // 2. Tokenization Demo
    // ========================================
    println!("2️⃣  Tokenization\n");
    
    let test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
    ];
    
    for text in test_texts {
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);
        println!("   \"{}\"", text);
        println!("   → Tokens: {:?}", &tokens[..tokens.len().min(10)]);
        println!("   → Decoded: \"{}\"", decoded);
        println!();
    }
    
    // ========================================
    // 3. Forward Pass Demo
    // ========================================
    println!("3️⃣  Forward Pass\n");
    
    let input_text = "Hello, world!";
    let input_tokens = tokenizer.encode(input_text);
    let input_ids: Vec<i64> = input_tokens.iter().map(|&x| x as i64).collect();
    let seq_len = input_ids.len();
    
    println!("   Input: \"{}\"", input_text);
    println!("   Token IDs: {:?}", input_ids);
    
    let input_tensor: Tensor<Backend, 2, Int> = Tensor::from_data(
        TensorData::new(input_ids, [1, seq_len]),
        &device,
    );
    
    let logits = model.forward(input_tensor);
    let [batch, seq, vocab] = logits.dims();
    
    println!("   Output logits shape: [{}, {}, {}]", batch, seq, vocab);
    println!();
    
    // ========================================
    // 4. Text Generation Demo
    // ========================================
    println!("4️⃣  Text Generation (Random Weights)\n");
    
    let generator = TextGenerator::new(model, tokenizer.clone(), device.clone());
    
    let prompts = ["The", "Hello", "I think"];
    
    for prompt in prompts {
        println!("   Prompt: \"{}\"", prompt);
        
        // Greedy
        let greedy_output = generator.generate_greedy(prompt, 8);
        println!("   Greedy:  \"{}\"", greedy_output);
        
        // Sampling
        let sampled_output = generator.generate(prompt, 8, 0.8, Some(50), None);
        println!("   Sampled: \"{}\"", sampled_output);
        println!();
    }
    
    // ========================================
    // 5. Dataset Demo
    // ========================================
    println!("5️⃣  Dataset Loading\n");
    
    let dataset_config = TextDatasetConfig {
        seq_length: 64,
        add_eos: true,
    };
    
    let demo_dataset = TextDataset::demo(&tokenizer, &dataset_config);
    println!("   Demo dataset size: {} samples", demo_dataset.len());
    
    // Show a sample
    // Access first sample from the dataset
    use burn::data::dataset::Dataset;
    if let Some(sample) = Dataset::get(&demo_dataset, 0) {
        let show_len = sample.input_ids.len().min(5);
        println!("   Sample input IDs: {:?}...", &sample.input_ids[..show_len]);
        let show_len = sample.target_ids.len().min(5);
        println!("   Sample target IDs: {:?}...", &sample.target_ids[..show_len]);
    }
    println!();
    
    // ========================================
    // 6. Evaluation Demo
    // ========================================
    println!("6️⃣  Model Evaluation\n");
    
    // Need to create a fresh model for evaluation (non-training)
    let eval_model: GPT2<Backend> = GPT2::new(&config, &device);
    let metrics = evaluate_model(&eval_model, &demo_dataset, 4, &device);
    
    println!("   Metrics on demo dataset:");
    println!("   ├─ Loss: {:.4}", metrics.loss);
    println!("   ├─ Perplexity: {:.2}", metrics.perplexity);
    println!("   └─ Accuracy: {:.2}%", metrics.accuracy * 100.0);
    println!();
    
    // ========================================
    // Summary
    // ========================================
    println!("{}", "═".repeat(50));
    println!();
    println!("✓ Demo complete!");
    println!();
    println!("Next steps:");
    println!("  1. Download training data:");
    println!("     cd utils && python download_nemotron.py --datasets demo wikitext");
    println!();
    println!("  2. Train the model:");
    println!("     cargo run --bin train -- --data data/wikitext2_train.jsonl");
    println!();
    println!("  3. Generate text:");
    println!("     cargo run --bin generate -- --prompt \"Once upon a time\"");
    println!();
}

fn estimate_parameters(config: &GPT2Config) -> usize {
    let token_emb = config.vocab_size * config.d_model;
    let pos_emb = config.max_seq_len * config.d_model;
    let attn = 4 * config.d_model * config.d_model * config.n_layers;
    let mlp = 2 * config.d_model * config.d_ff * config.n_layers;
    let lm_head = config.d_model * config.vocab_size;
    token_emb + pos_emb + attn + mlp + lm_head
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

