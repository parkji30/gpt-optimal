use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Simple BPE tokenizer for GPT-2
/// Note: For production use, consider using the `tokenizers` crate
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Token to ID mapping
    vocab: HashMap<String, usize>,
    /// ID to token mapping
    id_to_token: HashMap<usize, String>,
    /// Byte pair merges
    merges: Vec<(String, String)>,
    /// Special tokens
    bos_token_id: usize,
    eos_token_id: usize,
    pad_token_id: usize,
}

impl Tokenizer {
    /// Create a new simple tokenizer
    /// For real GPT-2, you'd load the actual vocab.json and merges.txt
    pub fn new() -> Self {
        // Create a simple character-level tokenizer as fallback
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        
        // Add special tokens
        vocab.insert("<|endoftext|>".to_string(), 50256);
        vocab.insert("<|pad|>".to_string(), 50257);
        id_to_token.insert(50256, "<|endoftext|>".to_string());
        id_to_token.insert(50257, "<|pad|>".to_string());
        
        // Add basic ASCII characters and common tokens
        let mut idx = 0;
        for c in 0u8..=127 {
            let ch = char::from(c);
            if ch.is_ascii_graphic() || ch == ' ' || ch == '\n' || ch == '\t' {
                let s = ch.to_string();
                if !vocab.contains_key(&s) {
                    vocab.insert(s.clone(), idx);
                    id_to_token.insert(idx, s);
                    idx += 1;
                }
            }
        }
        
        // Add some common word tokens
        let common_words = [
            "the", "and", "is", "in", "to", "of", "a", "that", "it", "for",
            "on", "with", "as", "was", "are", "be", "at", "this", "have", "from",
            "The", "I", "you", "he", "she", "we", "they", "my", "your", "his",
            "her", "its", "our", "their", "what", "which", "who", "when", "where",
            "how", "why", "can", "will", "would", "could", "should", "may", "might",
            "must", "shall", "do", "does", "did", "has", "had", "been", "being",
            "have", "having", "get", "got", "gotten", "make", "made", "go", "went",
            "come", "came", "see", "saw", "know", "knew", "think", "thought",
            "say", "said", "tell", "told", "ask", "asked", "use", "used",
            "find", "found", "give", "gave", "take", "took", "want", "wanted",
            "look", "looked", "need", "needed", "feel", "felt", "try", "tried",
            "leave", "left", "call", "called", "keep", "kept", "let", "put",
            "seem", "seemed", "help", "helped", "show", "showed", "hear", "heard",
            "play", "played", "run", "ran", "move", "moved", "live", "lived",
            "believe", "believed", "hold", "held", "bring", "brought", "happen",
            "happened", "write", "wrote", "provide", "provided", "sit", "sat",
            "stand", "stood", "lose", "lost", "pay", "paid", "meet", "met",
            "include", "included", "continue", "continued", "set", "learn", "learned",
            "change", "changed", "lead", "led", "understand", "understood", "watch",
            "watched", "follow", "followed", "stop", "stopped", "create", "created",
            "speak", "spoke", "read", "allow", "allowed", "add", "added", "spend",
            "spent", "grow", "grew", "open", "opened", "walk", "walked", "win",
            "won", "offer", "offered", "remember", "remembered", "love", "loved",
            "consider", "considered", "appear", "appeared", "buy", "bought", "wait",
            "waited", "serve", "served", "die", "died", "send", "sent", "expect",
            "expected", "build", "built", "stay", "stayed", "fall", "fell", "cut",
            "reach", "reached", "kill", "killed", "remain", "remained",
        ];
        
        for word in common_words {
            if !vocab.contains_key(word) {
                vocab.insert(word.to_string(), idx);
                id_to_token.insert(idx, word.to_string());
                idx += 1;
            }
        }
        
        Self {
            vocab,
            id_to_token,
            merges: Vec::new(),
            bos_token_id: 50256,
            eos_token_id: 50256,
            pad_token_id: 50257,
        }
    }
    
    /// Load tokenizer from vocab and merges files
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> std::io::Result<Self> {
        let vocab_file = File::open(vocab_path)?;
        let vocab_str = std::io::read_to_string(vocab_file)?;
        let vocab: HashMap<String, usize> = serde_json::from_str(&vocab_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        let id_to_token: HashMap<usize, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        
        // Load merges
        let merges_file = File::open(merges_path)?;
        let reader = BufReader::new(merges_file);
        let mut merges = Vec::new();
        
        for line in reader.lines().skip(1) {
            // Skip header
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }
        
        Ok(Self {
            vocab,
            id_to_token,
            merges,
            bos_token_id: 50256,
            eos_token_id: 50256,
            pad_token_id: 50257,
        })
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        
        // Simple word-level tokenization with character fallback
        for word in text.split_inclusive(|c: char| c.is_whitespace() || c.is_ascii_punctuation()) {
            if let Some(&id) = self.vocab.get(word.trim()) {
                tokens.push(id);
            } else {
                // Fallback to character-level
                for c in word.chars() {
                    if let Some(&id) = self.vocab.get(&c.to_string()) {
                        tokens.push(id);
                    } else {
                        // Unknown character, use a default
                        tokens.push(0);
                    }
                }
            }
        }
        
        tokens
    }
    
    /// Decode token IDs to text
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|id| self.id_to_token.get(id))
            .cloned()
            .collect()
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len().max(50258) // At least GPT-2 vocab size
    }
    
    /// Get end of sequence token ID
    pub fn eos_token_id(&self) -> usize {
        self.eos_token_id
    }
    
    /// Get padding token ID
    pub fn pad_token_id(&self) -> usize {
        self.pad_token_id
    }
    
    /// Get beginning of sequence token ID
    pub fn bos_token_id(&self) -> usize {
        self.bos_token_id
    }
    
    /// Encode with padding to a fixed length
    pub fn encode_with_padding(&self, text: &str, max_length: usize) -> Vec<usize> {
        let mut tokens = self.encode(text);
        
        if tokens.len() > max_length {
            tokens.truncate(max_length);
        } else {
            while tokens.len() < max_length {
                tokens.push(self.pad_token_id);
            }
        }
        
        tokens
    }
    
    /// Create training examples from text (for causal LM)
    /// Returns (input_ids, target_ids) where target is shifted by 1
    pub fn create_training_example(&self, text: &str, seq_length: usize) -> Option<(Vec<usize>, Vec<usize>)> {
        let tokens = self.encode(text);
        
        if tokens.len() < 2 {
            return None;
        }
        
        // Take up to seq_length + 1 tokens (need +1 for target shift)
        let tokens: Vec<usize> = if tokens.len() > seq_length + 1 {
            tokens[..seq_length + 1].to_vec()
        } else {
            tokens
        };
        
        if tokens.len() < 2 {
            return None;
        }
        
        let input_ids = tokens[..tokens.len() - 1].to_vec();
        let target_ids = tokens[1..].to_vec();
        
        Some((input_ids, target_ids))
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}


