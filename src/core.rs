//! Core AI Engine module
//! 
//! Provides the main processing engine for the AI framework.

use crate::{Config, Result};
use crate::algorithms::{Algorithm, AlgorithmType, StaticAlgorithm, DynamicAlgorithm};
use crate::coordination::CoordinationSystem;
use crate::gpu::GpuAccelerator;
use crate::verification::ConsistencyVerifier;

/// Main AI processing engine
pub struct Engine {
    coordination: CoordinationSystem,
    gpu_accelerator: Option<GpuAccelerator>,
    verifier: ConsistencyVerifier,
    config: Config,
}

impl Engine {
    /// Create a new engine instance
    pub fn new(config: &Config) -> Self {
        let gpu_accelerator = if config.gpu_enabled {
            Some(GpuAccelerator::new())
        } else {
            None
        };

        Self {
            coordination: CoordinationSystem::new(config.coordination_mode),
            gpu_accelerator,
            verifier: ConsistencyVerifier::new(config.verification_enabled),
            config: config.clone(),
        }
    }

    /// Process input data through the AI pipeline
    pub fn process(&self, input: &[f32]) -> Result<Vec<f32>> {
        // Select appropriate algorithm based on configuration
        let algorithm = self.create_algorithm()?;
        
        // Process through coordination system
        let result = self.coordination.process(&*algorithm, input)?;
        
        // Apply GPU acceleration if available
        let accelerated_result = if let Some(ref gpu) = self.gpu_accelerator {
            gpu.accelerate(&result)?
        } else {
            result
        };
        
        // Verify consistency if enabled
        if self.config.verification_enabled {
            self.verifier.verify(&accelerated_result)?;
        }
        
        Ok(accelerated_result)
    }

    /// Create algorithm instance based on configuration
    fn create_algorithm(&self) -> Result<Box<dyn Algorithm>> {
        match self.config.algorithm_type {
            AlgorithmType::Static => Ok(Box::new(StaticAlgorithm::new())),
            AlgorithmType::Dynamic => Ok(Box::new(DynamicAlgorithm::new())),
            AlgorithmType::Variable => {
                // For now, fallback to static
                Ok(Box::new(StaticAlgorithm::new()))
            }
        }
    }

    /// Get engine statistics
    pub fn stats(&self) -> EngineStats {
        EngineStats {
            gpu_enabled: self.gpu_accelerator.is_some(),
            coordination_mode: self.coordination.mode(),
            verification_enabled: self.config.verification_enabled,
        }
    }
}

/// Engine statistics and status
#[derive(Debug, Clone)]
pub struct EngineStats {
    pub gpu_enabled: bool,
    pub coordination_mode: crate::coordination::CoordinationMode,
    pub verification_enabled: bool,
}