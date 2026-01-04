use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Real-time training metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub step: usize,
    pub total_steps: usize,
    pub train_loss: f64,
    pub valid_loss: Option<f64>,
    pub learning_rate: f64,
    pub samples_per_second: f64,
    pub elapsed_time: Duration,
    pub perplexity: f64,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            epoch: 0,
            step: 0,
            total_steps: 0,
            train_loss: 0.0,
            valid_loss: None,
            learning_rate: 0.0,
            samples_per_second: 0.0,
            elapsed_time: Duration::ZERO,
            perplexity: 0.0,
        }
    }
}

/// Loss history for plotting
#[derive(Debug, Clone)]
pub struct LossHistory {
    pub train_losses: Vec<(usize, f64)>,
    pub valid_losses: Vec<(usize, f64)>,
}

impl Default for LossHistory {
    fn default() -> Self {
        Self {
            train_losses: Vec::new(),
            valid_losses: Vec::new(),
        }
    }
}

/// Simple terminal dashboard for training visualization
pub struct TrainingDashboard {
    metrics: Arc<Mutex<TrainingMetrics>>,
    loss_history: Arc<Mutex<LossHistory>>,
    start_time: Instant,
    last_update: Instant,
    update_interval: Duration,
}

impl TrainingDashboard {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(TrainingMetrics::default())),
            loss_history: Arc::new(Mutex::new(LossHistory::default())),
            start_time: Instant::now(),
            last_update: Instant::now(),
            update_interval: Duration::from_millis(100),
        }
    }
    
    /// Update training metrics
    pub fn update(&mut self, metrics: TrainingMetrics) {
        let mut m = self.metrics.lock().unwrap();
        *m = metrics;
        
        // Update loss history
        let mut history = self.loss_history.lock().unwrap();
        history.train_losses.push((m.step, m.train_loss));
        if let Some(valid_loss) = m.valid_loss {
            history.valid_losses.push((m.step, valid_loss));
        }
        
        // Display if enough time has passed
        if self.last_update.elapsed() >= self.update_interval {
            self.display();
            self.last_update = Instant::now();
        }
    }
    
    /// Display the current dashboard
    pub fn display(&self) {
        let metrics = self.metrics.lock().unwrap();
        let elapsed = self.start_time.elapsed();
        
        // Clear line and print progress
        print!("\r");
        print!(
            "Epoch {}/{} | Step {}/{} | Loss: {:.4} | PPL: {:.2} | LR: {:.2e} | Speed: {:.1} samples/s | Time: {:02}:{:02}:{:02}",
            metrics.epoch + 1,
            10, // TODO: Get from config
            metrics.step,
            metrics.total_steps,
            metrics.train_loss,
            metrics.perplexity,
            metrics.learning_rate,
            metrics.samples_per_second,
            elapsed.as_secs() / 3600,
            (elapsed.as_secs() % 3600) / 60,
            elapsed.as_secs() % 60,
        );
        io::stdout().flush().unwrap();
    }
    
    /// Display a loss chart in the terminal (ASCII art)
    pub fn display_loss_chart(&self) {
        let history = self.loss_history.lock().unwrap();
        
        if history.train_losses.is_empty() {
            println!("\nNo training data yet.");
            return;
        }
        
        println!("\n\n=== Loss History ===");
        
        // Get min/max for scaling
        let losses: Vec<f64> = history.train_losses.iter().map(|(_, l)| *l).collect();
        let min_loss = losses.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let height = 10;
        let width = 60.min(losses.len());
        
        // Downsample if needed
        let step_size = (losses.len() as f64 / width as f64).ceil() as usize;
        let downsampled: Vec<f64> = losses
            .chunks(step_size)
            .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
            .collect();
        
        // Draw chart
        for row in 0..height {
            let threshold = max_loss - (max_loss - min_loss) * (row as f64 / height as f64);
            
            if row == 0 {
                print!("{:8.4} |", max_loss);
            } else if row == height - 1 {
                print!("{:8.4} |", min_loss);
            } else {
                print!("         |");
            }
            
            for &loss in &downsampled {
                if loss >= threshold {
                    print!("█");
                } else {
                    print!(" ");
                }
            }
            println!();
        }
        
        // X-axis
        print!("         +");
        for _ in 0..downsampled.len() {
            print!("-");
        }
        println!();
        println!("          Step 0 {} Step {}", " ".repeat(downsampled.len().saturating_sub(20)), history.train_losses.len());
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> TrainingMetrics {
        self.metrics.lock().unwrap().clone()
    }
    
    /// Get loss history
    pub fn get_loss_history(&self) -> LossHistory {
        self.loss_history.lock().unwrap().clone()
    }
    
    /// Reset the dashboard
    pub fn reset(&mut self) {
        *self.metrics.lock().unwrap() = TrainingMetrics::default();
        *self.loss_history.lock().unwrap() = LossHistory::default();
        self.start_time = Instant::now();
    }
    
    /// Finish training display
    pub fn finish(&self) {
        println!("\n");
        self.display_loss_chart();
        
        let metrics = self.metrics.lock().unwrap();
        let elapsed = self.start_time.elapsed();
        
        println!("\n=== Training Complete ===");
        println!("Total time: {:02}:{:02}:{:02}", 
            elapsed.as_secs() / 3600,
            (elapsed.as_secs() % 3600) / 60,
            elapsed.as_secs() % 60
        );
        println!("Final train loss: {:.4}", metrics.train_loss);
        println!("Final perplexity: {:.2}", metrics.perplexity);
        if let Some(valid_loss) = metrics.valid_loss {
            println!("Final valid loss: {:.4}", valid_loss);
        }
    }
}

impl Default for TrainingDashboard {
    fn default() -> Self {
        Self::new()
    }
}

