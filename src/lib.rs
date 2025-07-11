//! Loneboth AI Framework
//! 
//! A modular AI framework supporting individual and group coordination,
//! static and dynamic algorithms, GPU acceleration, and consistency verification.

pub mod core;
pub mod algorithms;
pub mod coordination;
pub mod gpu;
pub mod verification;

pub use core::Engine;
pub use algorithms::{Algorithm, AlgorithmType};
pub use coordination::{CoordinationMode, CoordinationSystem};

/// Main result type for the framework
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Framework configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// Enable GPU acceleration
    pub gpu_enabled: bool,
    /// Coordination mode
    pub coordination_mode: CoordinationMode,
    /// Consistency verification enabled
    pub verification_enabled: bool,
    /// Algorithm type preference
    pub algorithm_type: AlgorithmType,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gpu_enabled: true,
            coordination_mode: CoordinationMode::Individual,
            verification_enabled: true,
            algorithm_type: AlgorithmType::Static,
        }
    }
}

/// Main entry point for the framework
pub struct LonebothAI {
    engine: Engine,
    config: Config,
}

impl LonebothAI {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: Config) -> Self {
        Self {
            engine: Engine::new(&config),
            config,
        }
    }

    /// Process input data through the AI framework
    pub fn process(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.engine.process(input)
    }

    /// Get current configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
}

impl Default for LonebothAI {
    fn default() -> Self {
        Self::new()
    }
}