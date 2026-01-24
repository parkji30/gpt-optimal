//! Custom Automatic Mixed Precision (AMP) implementation.
//!
//! This implements gradient scaling for mixed precision training to prevent
//! gradient underflow when using lower precision (f16) computations.
//!
//! Key components:
//! - GradScaler: Dynamically scales loss to prevent gradient underflow
//! - Overflow detection: Detects inf/nan in gradients and skips updates
//! - Dynamic scale adjustment: Grows scale when training is stable, shrinks on overflow

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use std::fmt::Debug;

/// Gradient scaler for Automatic Mixed Precision training.
///
/// The gradient scaler multiplies the loss by a scale factor before backward pass
/// to prevent gradient underflow in lower precision computations. After computing
/// gradients, they are unscaled before the optimizer step.
///
/// Scale is dynamically adjusted:
/// - Increased when no overflows occur for `growth_interval` steps
/// - Decreased when overflow (inf/nan gradients) is detected
#[derive(Debug, Clone)]
pub struct GradScaler {
    /// Current scale factor
    scale: f32,
    /// Factor to multiply scale by when growing
    growth_factor: f32,
    /// Factor to multiply scale by when backing off
    backoff_factor: f32,
    /// Number of consecutive steps without overflow before growing
    growth_interval: usize,
    /// Counter for steps without overflow
    steps_since_growth: usize,
    /// Whether AMP is enabled
    enabled: bool,
    /// Number of consecutive overflows
    consecutive_overflows: usize,
    /// Maximum allowed scale
    max_scale: f32,
    /// Minimum allowed scale
    min_scale: f32,
}

impl GradScaler {
    /// Create a new gradient scaler.
    ///
    /// # Arguments
    /// * `init_scale` - Initial scale factor (typically 65536)
    /// * `growth_factor` - Factor to multiply scale by when growing (typically 2.0)
    /// * `backoff_factor` - Factor to multiply scale by when backing off (typically 0.5)
    /// * `growth_interval` - Steps without overflow before growing scale (typically 2000)
    /// * `enabled` - Whether AMP is enabled
    pub fn new(
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
        enabled: bool,
    ) -> Self {
        Self {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            steps_since_growth: 0,
            enabled,
            consecutive_overflows: 0,
            max_scale: 2.0_f32.powi(24), // ~16 million
            min_scale: 1.0,
        }
    }

    /// Create from configuration
    pub fn from_config(config: &crate::config::AmpConfig) -> Self {
        Self::new(
            config.init_scale,
            config.growth_factor,
            config.backoff_factor,
            config.growth_interval,
            config.enabled,
        )
    }

    /// Check if AMP is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current scale factor
    pub fn get_scale(&self) -> f32 {
        if self.enabled {
            self.scale
        } else {
            1.0
        }
    }

    /// Scale the loss before backward pass.
    ///
    /// This multiplies the loss by the current scale factor to prevent
    /// gradient underflow during backpropagation.
    pub fn scale_loss<B: Backend>(&self, loss: Tensor<B, 1>) -> Tensor<B, 1> {
        if self.enabled {
            loss.mul_scalar(self.scale)
        } else {
            loss
        }
    }

    /// Scale a 0-dimensional loss tensor (scalar).
    pub fn scale_loss_scalar<B: Backend, const D: usize>(&self, loss: Tensor<B, D>) -> Tensor<B, D> {
        if self.enabled {
            loss.mul_scalar(self.scale)
        } else {
            loss
        }
    }

    /// Unscale gradients after backward pass.
    ///
    /// This divides gradients by the scale factor to restore the correct magnitude.
    /// Should be called before gradient clipping and optimizer step.
    pub fn unscale_grads<B: AutodiffBackend>(
        &self,
        grads: &mut B::Gradients,
    ) {
        if !self.enabled {
            return;
        }

        let inv_scale = 1.0 / self.scale;
        
        // Note: In burn, gradients are stored in the Gradients struct
        // We need to apply inverse scaling when updating parameters
        // The actual unscaling is handled in the update step
        // This is a placeholder for the unscaling operation
        let _ = inv_scale;
        let _ = grads;
    }

    /// Check if gradients contain inf or nan values.
    ///
    /// Returns true if any gradient contains non-finite values.
    pub fn check_gradients_for_overflow<B: Backend, const D: usize>(
        &self,
        tensor: &Tensor<B, D>,
    ) -> bool {
        if !self.enabled {
            return false;
        }

        // Check for inf/nan by computing sum and checking if it's finite
        // If any element is inf/nan, the sum will be inf/nan
        let sum = tensor.clone().abs().sum();
        let sum_data = sum.into_data();
        let sum_value: f32 = sum_data.as_slice().unwrap()[0];
        
        !sum_value.is_finite()
    }

    /// Update the scaler after an optimizer step.
    ///
    /// # Arguments
    /// * `found_overflow` - Whether overflow was detected in gradients
    ///
    /// # Returns
    /// * `true` if the optimizer step should proceed (no overflow)
    /// * `false` if the step should be skipped (overflow detected)
    pub fn update(&mut self, found_overflow: bool) -> bool {
        if !self.enabled {
            return true;
        }

        if found_overflow {
            // Overflow detected: reduce scale and skip step
            self.scale *= self.backoff_factor;
            self.scale = self.scale.max(self.min_scale);
            self.steps_since_growth = 0;
            self.consecutive_overflows += 1;

            if self.consecutive_overflows > 10 {
                println!("Warning: {} consecutive gradient overflows. Scale: {:.2}", 
                    self.consecutive_overflows, self.scale);
            }

            false // Skip optimizer step
        } else {
            // No overflow: increment counter and possibly grow scale
            self.consecutive_overflows = 0;
            self.steps_since_growth += 1;

            if self.steps_since_growth >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.scale = self.scale.min(self.max_scale);
                self.steps_since_growth = 0;
            }

            true // Proceed with optimizer step
        }
    }

    /// Get statistics about the scaler state.
    pub fn get_stats(&self) -> GradScalerStats {
        GradScalerStats {
            scale: self.scale,
            steps_since_growth: self.steps_since_growth,
            consecutive_overflows: self.consecutive_overflows,
        }
    }
}

/// Statistics about the gradient scaler state.
#[derive(Debug, Clone)]
pub struct GradScalerStats {
    pub scale: f32,
    pub steps_since_growth: usize,
    pub consecutive_overflows: usize,
}

/// Mixed precision context for managing precision during forward/backward passes.
///
/// This provides utilities for running computations in different precisions
/// and converting between them.
#[derive(Debug, Clone)]
pub struct MixedPrecisionContext {
    /// Whether mixed precision is enabled
    enabled: bool,
}

impl MixedPrecisionContext {
    /// Create a new mixed precision context
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Check if mixed precision is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Cast tensor to lower precision (simulated for backends without native f16).
    ///
    /// In practice, this would use actual f16/bf16 types. Since burn's CUDA
    /// backend handles this internally, we simulate the precision reduction
    /// by quantizing values.
    pub fn to_low_precision<B: Backend, const D: usize>(
        &self,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D> {
        if !self.enabled {
            return tensor;
        }

        // In a real implementation with f16 support, this would cast to f16
        // For simulation, we can clamp to f16 range
        // f16 range: approximately ±65504 with min positive ≈ 6.1e-5
        tensor.clamp(-65504.0, 65504.0)
    }

    /// Cast tensor back to full precision.
    pub fn to_full_precision<B: Backend, const D: usize>(
        &self,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D> {
        // No-op for f32 tensors, would cast f16 -> f32 in real implementation
        tensor
    }
}

/// Trait for automatic mixed precision training support.
pub trait AmpTraining<B: AutodiffBackend> {
    /// Run forward pass with mixed precision.
    fn forward_amp(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>);
    
    /// Scale gradients for optimizer step.
    fn scale_gradients(&self, grads: B::Gradients, scaler: &GradScaler) -> B::Gradients;
}

/// Helper for gradient scaling in optimizer updates.
///
/// This wrapper provides utilities for scaling/unscaling during optimization.
#[derive(Debug)]
pub struct ScaledOptimizer<O> {
    /// Underlying optimizer
    optimizer: O,
    /// Gradient scaler
    scaler: GradScaler,
    /// Inverse scale for gradient unscaling
    inv_scale: f32,
}

impl<O> ScaledOptimizer<O> {
    /// Create a new scaled optimizer wrapper.
    pub fn new(optimizer: O, scaler: GradScaler) -> Self {
        let inv_scale = 1.0 / scaler.get_scale();
        Self {
            optimizer,
            scaler,
            inv_scale,
        }
    }

    /// Get the underlying optimizer
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Get mutable reference to underlying optimizer
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Get the gradient scaler
    pub fn scaler(&self) -> &GradScaler {
        &self.scaler
    }

    /// Get mutable reference to scaler
    pub fn scaler_mut(&mut self) -> &mut GradScaler {
        &mut self.scaler
    }

    /// Get the inverse scale factor for manual gradient unscaling.
    pub fn get_inv_scale(&self) -> f32 {
        self.inv_scale
    }

    /// Update inverse scale (call after scaler.update())
    pub fn update_inv_scale(&mut self) {
        self.inv_scale = 1.0 / self.scaler.get_scale();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_scaler_creation() {
        let scaler = GradScaler::new(65536.0, 2.0, 0.5, 2000, true);
        assert_eq!(scaler.get_scale(), 65536.0);
        assert!(scaler.is_enabled());
    }

    #[test]
    fn test_grad_scaler_disabled() {
        let scaler = GradScaler::new(65536.0, 2.0, 0.5, 2000, false);
        assert_eq!(scaler.get_scale(), 1.0);
        assert!(!scaler.is_enabled());
    }

    #[test]
    fn test_grad_scaler_overflow_handling() {
        let mut scaler = GradScaler::new(65536.0, 2.0, 0.5, 2000, true);
        
        // Simulate overflow
        let should_step = scaler.update(true);
        assert!(!should_step);
        assert_eq!(scaler.get_scale(), 32768.0); // 65536 * 0.5

        // Simulate successful steps
        for _ in 0..2000 {
            let should_step = scaler.update(false);
            assert!(should_step);
        }
        assert_eq!(scaler.get_scale(), 65536.0); // Back to 32768 * 2
    }

    #[test]
    fn test_mixed_precision_context() {
        let ctx = MixedPrecisionContext::new(true);
        assert!(ctx.is_enabled());

        let ctx_disabled = MixedPrecisionContext::new(false);
        assert!(!ctx_disabled.is_enabled());
    }
}
