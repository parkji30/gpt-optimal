use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::CrossEntropyLossConfig,
    tensor::backend::Backend,
};

use crate::{
    data::{TextBatcher, TextDataset},
    data::dataset::TextItem,
    model::GPT2,
    tokenizer::Tokenizer,
};

/// Evaluation metrics
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    /// Average loss
    pub loss: f64,
    /// Perplexity
    pub perplexity: f64,
    /// Accuracy (next token prediction)
    pub accuracy: f64,
    /// Number of samples evaluated
    pub num_samples: usize,
}

impl EvalMetrics {
    pub fn new() -> Self {
        Self {
            loss: 0.0,
            perplexity: 0.0,
            accuracy: 0.0,
            num_samples: 0,
        }
    }
}

impl Default for EvalMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate model on a dataset
pub fn evaluate_model<B: Backend>(
    model: &GPT2<B>,
    dataset: &TextDataset,
    batch_size: usize,
    device: &B::Device,
) -> EvalMetrics {
    use crate::data::TextBatch;
    
    let tokenizer = Tokenizer::new();
    let pad_token_id = tokenizer.pad_token_id();
    
    let batcher: TextBatcher<B> = TextBatcher::new(device.clone(), pad_token_id);
    let dataloader = DataLoaderBuilder::<B, TextItem, TextBatch<B>>::new(batcher)
        .batch_size(batch_size)
        .build(dataset.clone());
    
    let mut total_loss = 0.0f64;
    let mut total_correct = 0usize;
    let mut total_tokens = 0usize;
    let mut num_batches = 0usize;
    
    for batch in dataloader.iter() {
        let logits = model.forward(batch.input_ids.clone());
        let [batch_size_actual, seq_len, vocab_size] = logits.dims();
        
        // Calculate loss
        let logits_flat = logits.clone().reshape([batch_size_actual * seq_len, vocab_size]);
        let targets_flat = batch.target_ids.clone().reshape([batch_size_actual * seq_len]);
        
        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_token_id]))
            .init(device)
            .forward(logits_flat.clone(), targets_flat.clone());
        
        // Extract loss value
        let loss_data = loss.into_data();
        let loss_value: f32 = loss_data.to_vec::<f32>().unwrap()[0];
        total_loss += loss_value as f64;
        
        // Calculate accuracy: compare predictions to targets
        let predictions = logits_flat.argmax(1); // [batch * seq_len]
        
        // Get prediction and target data
        let pred_data = predictions.into_data();
        let target_data = targets_flat.clone().into_data();
        
        let pred_vec: Vec<i32> = pred_data.to_vec().unwrap();
        let target_vec: Vec<i32> = target_data.to_vec().unwrap();
        
        // Count correct predictions (ignoring padding)
        for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
            if *target != pad_token_id as i32 {
                total_tokens += 1;
                if pred == target {
                    total_correct += 1;
                }
            }
        }
        
        num_batches += 1;
    }
    
    let avg_loss = if num_batches > 0 {
        total_loss / num_batches as f64
    } else {
        0.0
    };
    
    let accuracy = if total_tokens > 0 {
        total_correct as f64 / total_tokens as f64
    } else {
        0.0
    };
    
    EvalMetrics {
        loss: avg_loss,
        perplexity: avg_loss.exp(),
        accuracy,
        num_samples: dataset.len(),
    }
}

/// Evaluate and print results
pub fn evaluate_and_print<B: Backend>(
    model: &GPT2<B>,
    dataset: &TextDataset,
    batch_size: usize,
    device: &B::Device,
    dataset_name: &str,
) {
    println!("\n=== Evaluating on {} ===", dataset_name);
    let metrics = evaluate_model(model, dataset, batch_size, device);
    
    println!("  Samples: {}", metrics.num_samples);
    println!("  Loss: {:.4}", metrics.loss);
    println!("  Perplexity: {:.4}", metrics.perplexity);
    println!("  Accuracy: {:.2}%", metrics.accuracy * 100.0);
}
