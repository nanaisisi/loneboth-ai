//! LoneBoth AI Framework
//! 
//! A comprehensive AI framework for algorithm coordination and GPU-accelerated machine learning.

use std::error::Error;
use log::info;

pub mod algorithm;
pub mod coordination;
pub mod gpu;
pub mod runtime;
pub mod verification;

pub mod prelude {
    pub use crate::algorithm::{Algorithm, AlgorithmResult, AlgorithmType, AlgorithmRegistry, StaticAlgorithm, DynamicAlgorithm, FuzzyAlgorithm};
    pub use crate::coordination::{Coordinator, IndividualCoordinator, GroupCoordinator, CoordinationManager, CoordinationType, CoordinationResult};
    pub use crate::gpu::{ONNXExecutor, ExecutionProvider, Tensor, GpuDevice, GpuAccelerator, OnnxRuntimeExecutor};
    pub use crate::runtime::{DynamicLoader, RuntimeManager, AlgorithmMetadata};
    pub use crate::verification::{VerificationSystem, Verifier, VerificationResult};
    pub use crate::LoneBothAI;
}

/// Main framework structure
pub struct LoneBothAI {
    initialized: bool,
}

impl LoneBothAI {
    /// Create a new LoneBoth AI framework instance
    pub fn new() -> Result<Self, Box<dyn Error>> {
        // Only initialize env_logger if it hasn't been initialized yet
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "info");
        }
        let _ = env_logger::try_init();
        
        info!("Initializing LoneBoth AI Framework");
        
        Ok(Self {
            initialized: true,
        })
    }
    
    /// Check if framework is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl Default for LoneBothAI {
    fn default() -> Self {
        Self::new().expect("Failed to initialize LoneBoth AI Framework")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_initialization() {
        let framework = LoneBothAI::new().unwrap();
        assert!(framework.is_initialized());
    }

    #[test]
    fn test_default_initialization() {
        let framework = LoneBothAI::default();
        assert!(framework.is_initialized());
    }
}