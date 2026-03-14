//! Character-level dataset for GPT-2 training.

use burn::prelude::*;
use rand::seq::SliceRandom;
use std::path::Path;

/// Character-level dataset for training.
///
/// Converts text to byte sequences and provides (input, target) pairs
/// where target is input shifted by one position.
#[derive(Debug, Clone)]
pub struct CharDataset {
    /// Raw data as byte indices
    data: Vec<i64>,
    /// Block size (context length)
    block_size: usize,
    /// Valid starting indices for samples
    valid_indices: Vec<usize>,
}

impl CharDataset {
    /// Create a new character dataset from text.
    ///
    /// # Arguments
    /// * `text` - The text to use for training
    /// * `block_size` - The context length for each sample
    pub fn new(text: &str, block_size: usize) -> Self {
        // Convert characters to bytes (0-255)
        let data: Vec<i64> = text.bytes().map(|b| b as i64).collect();
        
        // Valid indices: any position where we can get block_size + 1 characters
        let valid_indices: Vec<usize> = (0..data.len().saturating_sub(block_size))
            .collect();

        Self {
            data,
            block_size,
            valid_indices,
        }
    }

    /// Load dataset from a file.
    pub fn from_file(path: &Path, block_size: usize) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(Self::new(&text, block_size))
    }

    /// Get the number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.valid_indices.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.valid_indices.is_empty()
    }

    /// Get a single sample by index.
    ///
    /// # Returns
    /// * (input, target) where both are vectors of block_size token indices
    pub fn get(&self, idx: usize) -> (Vec<i64>, Vec<i64>) {
        let start = self.valid_indices[idx];
        let x = self.data[start..start + self.block_size].to_vec();
        let y = self.data[start + 1..start + self.block_size + 1].to_vec();
        (x, y)
    }

    /// Get a batch of samples as tensors.
    ///
    /// # Arguments
    /// * `indices` - Indices of samples to include in the batch
    /// * `device` - Device to create tensors on
    ///
    /// # Returns
    /// * (inputs, targets) tensors of shape [batch_size, block_size]
    pub fn get_batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let batch_size = indices.len();
        
        let mut x_data = Vec::with_capacity(batch_size * self.block_size);
        let mut y_data = Vec::with_capacity(batch_size * self.block_size);

        for &idx in indices {
            let (x, y) = self.get(idx);
            x_data.extend(x);
            y_data.extend(y);
        }

        let x = Tensor::<B, 1, Int>::from_data(x_data.as_slice(), device)
            .reshape([batch_size, self.block_size]);
        let y = Tensor::<B, 1, Int>::from_data(y_data.as_slice(), device)
            .reshape([batch_size, self.block_size]);

        (x, y)
    }
}

/// Batch iterator for efficient training.
pub struct BatchIterator {
    /// All valid indices
    indices: Vec<usize>,
    /// Current position in indices
    position: usize,
    /// Batch size
    batch_size: usize,
    /// Whether to shuffle indices
    shuffle: bool,
}

impl BatchIterator {
    /// Create a new batch iterator.
    pub fn new(dataset_len: usize, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..dataset_len).collect();
        Self {
            indices,
            position: 0,
            batch_size,
            shuffle,
        }
    }

    /// Reset the iterator, optionally shuffling indices.
    pub fn reset(&mut self) {
        self.position = 0;
        if self.shuffle {
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get the next batch of indices.
    ///
    /// Returns None when epoch is complete.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.indices.len());
        
        // Skip incomplete batches at the end
        if end - self.position < self.batch_size && end < self.indices.len() {
            return None;
        }

        let batch = self.indices[self.position..end].to_vec();
        self.position = end;

        // Skip if batch is smaller than batch_size (last incomplete batch)
        if batch.len() < self.batch_size {
            return None;
        }

        Some(batch)
    }

    /// Get a random batch of indices (for validation).
    pub fn random_batch(&self) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut indices = self.indices.clone();
        indices.shuffle(&mut rng);
        batch.extend(indices.into_iter().take(self.batch_size));
        batch
    }
}

/// Data loader combining dataset and batch iterator.
pub struct DataLoader {
    /// The dataset
    dataset: CharDataset,
    /// Batch iterator
    iterator: BatchIterator,
    /// Batch size
    batch_size: usize,
}

impl DataLoader {
    /// Create a new data loader.
    pub fn new(dataset: CharDataset, batch_size: usize, shuffle: bool) -> Self {
        let iterator = BatchIterator::new(dataset.len(), batch_size, shuffle);
        Self {
            dataset,
            iterator,
            batch_size,
        }
    }

    /// Get the underlying dataset.
    pub fn dataset(&self) -> &CharDataset {
        &self.dataset
    }

    /// Reset the data loader for a new epoch.
    pub fn reset(&mut self) {
        self.iterator.reset();
    }

    /// Get the next batch of tensors.
    pub fn next_batch<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Option<(Tensor<B, 2, Int>, Tensor<B, 2, Int>)> {
        self.iterator.next_batch().map(|indices| {
            self.dataset.get_batch::<B>(&indices, device)
        })
    }

    /// Get a random batch (useful for evaluation).
    pub fn random_batch<B: Backend>(
        &self,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let indices = self.iterator.random_batch();
        self.dataset.get_batch::<B>(&indices, device)
    }

    /// Get the number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        self.dataset.len() / self.batch_size
    }

    /// Get the total number of samples.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_char_dataset_creation() {
        let text = "Hello, World!";
        let dataset = CharDataset::new(text, 4);
        
        assert_eq!(dataset.len(), text.len() - 4);
    }

    #[test]
    fn test_char_dataset_get() {
        let text = "abcdefgh";
        let dataset = CharDataset::new(text, 4);
        
        let (x, y) = dataset.get(0);
        assert_eq!(x, vec![97, 98, 99, 100]); // "abcd"
        assert_eq!(y, vec![98, 99, 100, 101]); // "bcde"
    }

    #[test]
    fn test_char_dataset_get_batch() {
        let device = Default::default();
        let text = "abcdefghijklmnop";
        let dataset = CharDataset::new(text, 4);
        
        let (x, y) = dataset.get_batch::<TestBackend>(&[0, 1], &device);
        
        assert_eq!(x.dims(), [2, 4]);
        assert_eq!(y.dims(), [2, 4]);
    }

    #[test]
    fn test_batch_iterator() {
        let mut iter = BatchIterator::new(100, 10, false);
        
        let mut count = 0;
        while let Some(batch) = iter.next_batch() {
            assert_eq!(batch.len(), 10);
            count += 1;
        }
        
        assert_eq!(count, 10);
    }

    #[test]
    fn test_data_loader() {
        let device = Default::default();
        let text = "a]".repeat(100);
        let dataset = CharDataset::new(&text, 4);
        let mut loader = DataLoader::new(dataset, 8, true);
        
        let mut count = 0;
        while let Some((x, y)) = loader.next_batch::<TestBackend>(&device) {
            assert_eq!(x.dims()[0], 8);
            assert_eq!(y.dims()[0], 8);
            count += 1;
        }
        
        assert!(count > 0);
    }
}
