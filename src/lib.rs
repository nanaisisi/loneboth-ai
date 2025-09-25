//! Loneboth AI Framework - Burn-based implementation
//! 
//! An action-oriented AI framework focusing on consistent training-inference pipelines,
//! environmental adaptation, and structural-relational execution patterns.

use burn::backend::{Autodiff, Candle, Wgpu};
use burn::tensor::{Tensor, Device};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

pub mod core;
pub mod environment;
pub mod behavior;
pub mod training;
pub mod inference;
pub mod adaptation;

pub use core::{ActionExecutor, StructuralContext};
pub use environment::{Environment, EnvironmentState, Observation};
pub use behavior::{BehaviorPattern, ActionSpace, PolicyNetwork};
pub use training::{TrainingPipeline, AdaptationLearner};
pub use inference::{InferenceEngine, DecisionMaker};

/// Backend type aliases for clarity
pub type Backend = Candle<f32>;
pub type AutodiffBackend = Autodiff<Backend>;

/// Main result type with enhanced error context
pub type LonebothResult<T> = Result<T>;

/// Core configuration defining the framework's operational parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Device configuration for computation
    pub device: DeviceConfig,
    /// Environment interaction parameters
    pub environment: EnvironmentConfig,
    /// Behavioral adaptation settings
    pub adaptation: AdaptationConfig,
    /// Training configuration
    pub training: TrainingConfig,
    /// Inference optimization
    pub inference: InferenceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub use_gpu: bool,
    pub backend_preference: BackendType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendType {
    Candle,
    Wgpu,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    pub observation_dimension: usize,
    pub action_dimension: usize,
    pub temporal_context_length: usize,
    pub adaptation_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    pub learning_rate: f64,
    pub adaptation_frequency: usize,
    pub structural_weight: f32,
    pub relational_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub validation_split: f32,
    pub early_stopping: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub batch_optimization: bool,
    pub cache_policy_states: bool,
    pub real_time_adaptation: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig {
                use_gpu: true,
                backend_preference: BackendType::Auto,
            },
            environment: EnvironmentConfig {
                observation_dimension: 128,
                action_dimension: 32,
                temporal_context_length: 10,
                adaptation_threshold: 0.1,
            },
            adaptation: AdaptationConfig {
                learning_rate: 1e-3,
                adaptation_frequency: 100,
                structural_weight: 0.6,
                relational_weight: 0.4,
            },
            training: TrainingConfig {
                batch_size: 32,
                epochs: 100,
                validation_split: 0.2,
                early_stopping: true,
            },
            inference: InferenceConfig {
                batch_optimization: true,
                cache_policy_states: true,
                real_time_adaptation: true,
            },
        }
    }
}

/// Central orchestrator for the Loneboth AI system
/// Integrates training and inference in a unified, action-oriented framework
pub struct LonebothAI<B: burn::prelude::Backend> {
    config: SystemConfig,
    executor: ActionExecutor<B>,
    environment: Environment<B>,
    training_pipeline: TrainingPipeline<B>,
    inference_engine: InferenceEngine<B>,
    device: Device<B>,
}

impl<B: burn::prelude::Backend> LonebothAI<B> {
    /// Initialize with default configuration
    pub fn new(device: Device<B>) -> LonebothResult<Self> {
        Self::with_config(SystemConfig::default(), device)
    }

    /// Initialize with custom configuration
    pub fn with_config(config: SystemConfig, device: Device<B>) -> LonebothResult<Self> {
        info!("Initializing Loneboth AI with config: {:?}", config);

        let executor = ActionExecutor::new(&config, device.clone())?;
        let environment = Environment::new(&config.environment)?;
        let training_pipeline = TrainingPipeline::new(&config.training, &config.adaptation, device.clone())?;
        let inference_engine = InferenceEngine::new(&config.inference, device.clone())?;

        Ok(Self {
            config,
            executor,
            environment,
            training_pipeline,
            inference_engine,
            device,
        })
    }

    /// Execute action in environment with unified training-inference approach
    pub async fn execute_action(&mut self, observation: Tensor<B, 2>) -> LonebothResult<Tensor<B, 2>> {
        // Update environmental context
        let env_state = self.environment.update_state(observation).await?;
        
        // Execute action through unified pipeline
        let action = self.executor.execute(&env_state).await?;
        
        // Adapt based on execution results if needed
        if self.should_adapt(&env_state)? {
            self.adapt_to_environment(&env_state, &action).await?;
        }
        
        Ok(action)
    }

    /// Train the system using collected experience
    pub async fn train(&mut self, experience_data: Vec<(Tensor<B, 2>, Tensor<B, 2>)>) -> LonebothResult<()> {
        info!("Starting training with {} experience samples", experience_data.len());
        
        let training_result = self.training_pipeline
            .train_behavioral_patterns(experience_data)
            .await?;
            
        // Update inference engine with trained patterns
        self.inference_engine
            .update_policy(training_result.policy_weights)
            .await?;
            
        info!("Training completed successfully");
        Ok(())
    }

    /// Perform inference on new observations
    pub async fn infer(&self, observations: Tensor<B, 2>) -> LonebothResult<Tensor<B, 2>> {
        self.inference_engine.predict(observations).await
    }

    /// Adapt system to current environmental conditions
    async fn adapt_to_environment(&mut self, env_state: &EnvironmentState<B>, action: &Tensor<B, 2>) -> LonebothResult<()> {
        warn!("Adapting to environmental changes");
        
        // Structural adaptation based on environment
        let structural_adjustment = self.environment
            .compute_structural_adaptation(env_state)
            .await?;
            
        // Relational adaptation based on action outcomes
        let relational_adjustment = self.executor
            .compute_relational_adaptation(action, env_state)
            .await?;
            
        // Apply combined adaptations
        self.training_pipeline
            .apply_adaptations(structural_adjustment, relational_adjustment)
            .await?;
            
        Ok(())
    }

    /// Determine if adaptation is needed based on environmental change
    fn should_adapt(&self, env_state: &EnvironmentState<B>) -> LonebothResult<bool> {
        let change_magnitude = env_state.change_magnitude();
        Ok(change_magnitude > self.config.environment.adaptation_threshold)
    }

    /// Get current system configuration
    pub fn config(&self) -> &SystemConfig {
        &self.config
    }

    /// Get system performance metrics
    pub fn get_metrics(&self) -> SystemMetrics {
        SystemMetrics {
            total_actions_executed: self.executor.total_executions(),
            training_iterations: self.training_pipeline.iterations(),
            inference_speed: self.inference_engine.average_latency(),
            adaptation_events: self.environment.adaptation_count(),
        }
    }
}

/// System performance and monitoring metrics
#[derive(Debug, Clone, Serialize)]
pub struct SystemMetrics {
    pub total_actions_executed: u64,
    pub training_iterations: u64,
    pub inference_speed: f64,
    pub adaptation_events: u64,
}