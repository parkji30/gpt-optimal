use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use rand::Rng;

use crate::model::GPT2;
use crate::tokenizer::Tokenizer;

/// Text generator for GPT-2
pub struct TextGenerator<B: Backend> {
    model: GPT2<B>,
    tokenizer: Tokenizer,
    device: B::Device,
}

impl<B: Backend> TextGenerator<B> {
    pub fn new(model: GPT2<B>, tokenizer: Tokenizer, device: B::Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }
    
    /// Generate text from a prompt
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
    ) -> String {
        // Encode prompt
        let input_ids = self.tokenizer.encode(prompt);
        if input_ids.is_empty() {
            return prompt.to_string();
        }
        
        let mut current_ids = input_ids;
        let mut rng = rand::thread_rng();
        
        for _ in 0..max_new_tokens {
            // Truncate if needed
            let max_seq_len = self.model.max_seq_len();
            let context_ids: Vec<i64> = if current_ids.len() > max_seq_len {
                current_ids[(current_ids.len() - max_seq_len)..]
                    .iter()
                    .map(|&x| x as i64)
                    .collect()
            } else {
                current_ids.iter().map(|&x| x as i64).collect()
            };
            
            let seq_len = context_ids.len();
            
            // Create input tensor
            let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(
                TensorData::new(context_ids, [1, seq_len]),
                &self.device,
            );
            
            // Get logits
            let logits = self.model.forward(input_tensor);
            let [_batch, seq_len_out, vocab_size] = logits.dims();
            
            // Get last token logits
            let last_logits = logits
                .slice([0..1, (seq_len_out - 1)..seq_len_out, 0..vocab_size])
                .reshape([vocab_size]);
            
            // Apply temperature
            let scaled_logits = last_logits / temperature;
            
            // Convert to probabilities
            let probs = burn::tensor::activation::softmax(scaled_logits, 0);
            
            // Sample next token
            let next_token = self.sample_token(&probs, top_k, top_p, &mut rng);
            
            // Check for EOS
            if next_token == self.tokenizer.eos_token_id() {
                break;
            }
            
            current_ids.push(next_token);
        }
        
        self.tokenizer.decode(&current_ids)
    }
    
    /// Sample a token from the probability distribution
    fn sample_token(
        &self,
        probs: &Tensor<B, 1>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        rng: &mut impl Rng,
    ) -> usize {
        let probs_data: Vec<f32> = probs.to_data().to_vec().unwrap();
        
        // Apply top-k filtering
        let mut indexed_probs: Vec<(usize, f32)> = probs_data
            .into_iter()
            .enumerate()
            .collect();
        
        // Sort by probability descending
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply top-k
        if let Some(k) = top_k {
            indexed_probs.truncate(k);
        }
        
        // Apply top-p (nucleus sampling)
        if let Some(p) = top_p {
            let mut cumsum = 0.0;
            let mut cutoff_idx = indexed_probs.len();
            for (i, (_, prob)) in indexed_probs.iter().enumerate() {
                cumsum += *prob as f64;
                if cumsum >= p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            indexed_probs.truncate(cutoff_idx);
        }
        
        // Renormalize
        let total: f32 = indexed_probs.iter().map(|(_, p)| p).sum();
        let normalized: Vec<(usize, f32)> = indexed_probs
            .into_iter()
            .map(|(idx, p)| (idx, p / total))
            .collect();
        
        // Sample
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (idx, prob) in normalized {
            cumsum += prob;
            if r <= cumsum {
                return idx;
            }
        }
        
        // Fallback to most likely
        0
    }
    
    /// Generate with greedy decoding (no sampling)
    pub fn generate_greedy(&self, prompt: &str, max_new_tokens: usize) -> String {
        let input_ids = self.tokenizer.encode(prompt);
        if input_ids.is_empty() {
            return prompt.to_string();
        }
        
        let mut current_ids = input_ids;
        
        for _ in 0..max_new_tokens {
            let max_seq_len = self.model.max_seq_len();
            let context_ids: Vec<i64> = if current_ids.len() > max_seq_len {
                current_ids[(current_ids.len() - max_seq_len)..]
                    .iter()
                    .map(|&x| x as i64)
                    .collect()
            } else {
                current_ids.iter().map(|&x| x as i64).collect()
            };
            
            let seq_len = context_ids.len();
            
            let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(
                TensorData::new(context_ids, [1, seq_len]),
                &self.device,
            );
            
            let logits = self.model.forward(input_tensor);
            let [_batch, seq_len_out, vocab_size] = logits.dims();
            
            let last_logits = logits
                .slice([0..1, (seq_len_out - 1)..seq_len_out, 0..vocab_size])
                .reshape([vocab_size]);
            
            // Greedy selection
            let next_token_data = last_logits.argmax(0).into_data();
            let next_token: usize = next_token_data.to_vec::<i32>().unwrap()[0] as usize;
            
            if next_token == self.tokenizer.eos_token_id() {
                break;
            }
            
            current_ids.push(next_token);
        }
        
        self.tokenizer.decode(&current_ids)
    }
}

