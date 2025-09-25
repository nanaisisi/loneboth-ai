//! Environment module for environmental state management and adaptation
//! 
//! Manages environmental observations, state transitions, and adaptation detection
//! with burn tensor operations for efficient computation.

use burn::prelude::*;
use burn::tensor::{Tensor, Device};
use crate::{LonebothResult, EnvironmentConfig};
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{info, debug, warn};

/// Environment system managing state, observations, and environmental dynamics
#[derive(Debug)]
pub struct Environment<B: Backend> {
    /// Current environmental state
    current_state: EnvironmentState<B>,
    /// Historical observations for temporal analysis
    observation_history: VecDeque<Tensor<B, 2>>,
    /// Environment configuration
    config: EnvironmentConfig,
    /// Adaptation event counter
    adaptation_count: u64,
    /// Device for tensor operations
    device: Device<B>,
}

/// Complete environmental state representation
#[derive(Debug, Clone)]
pub struct EnvironmentState<B: Backend> {
    /// Current observation tensor
    observation: Tensor<B, 2>,
    /// Temporal context from recent observations
    temporal_context: Tensor<B, 2>,
    /// Detected change magnitude
    change_magnitude: f32,
    /// State timestamp
    timestamp: u64,
}

/// Single observation from the environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Raw observation data
    pub data: Vec<f32>,
    /// Observation timestamp
    pub timestamp: u64,
    /// Optional metadata
    pub metadata: Option<ObservationMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationMetadata {
    /// Source of the observation
    pub source: String,
    /// Confidence in observation quality
    pub confidence: f32,
    /// Additional context
    pub context: Vec<(String, String)>,
}

impl<B: Backend> Environment<B> {
    pub fn new(config: &EnvironmentConfig) -> LonebothResult<Self> {
        info!("Initializing Environment with config: {:?}", config);
        
        let device = Device::default();
        let initial_observation = Tensor::zeros([1, config.observation_dimension], &device);
        let initial_temporal_context = Tensor::zeros([1, config.observation_dimension * config.temporal_context_length], &device);
        
        let current_state = EnvironmentState {
            observation: initial_observation,
            temporal_context: initial_temporal_context,
            change_magnitude: 0.0,
            timestamp: 0,
        };
        
        Ok(Self {
            current_state,
            observation_history: VecDeque::with_capacity(config.temporal_context_length),
            config: config.clone(),
            adaptation_count: 0,
            device,
        })
    }
    
    /// Update environmental state with new observation
    pub async fn update_state(&mut self, observation: Tensor<B, 2>) -> LonebothResult<&EnvironmentState<B>> {
        debug!("Updating environmental state");
        
        // Calculate change magnitude from previous state
        let change_magnitude = self.calculate_change_magnitude(&observation)?;
        
        // Update observation history
        self.observation_history.push_back(observation.clone());
        if self.observation_history.len() > self.config.temporal_context_length {
            self.observation_history.pop_front();
        }
        
        // Build temporal context from history
        let temporal_context = self.build_temporal_context()?;
        
        // Update current state
        self.current_state = EnvironmentState {
            observation,
            temporal_context,
            change_magnitude,
            timestamp: self.current_state.timestamp + 1,
        };
        
        debug!("Environmental state updated, change magnitude: {}", change_magnitude);
        Ok(&self.current_state)
    }
    
    /// Calculate magnitude of change from previous observation
    fn calculate_change_magnitude(&self, new_observation: &Tensor<B, 2>) -> LonebothResult<f32> {
        if self.current_state.timestamp == 0 {
            return Ok(0.0);
        }
        
        let previous_obs = &self.current_state.observation;
        let difference = new_observation.clone() - previous_obs.clone();
        let magnitude = difference.abs().mean().into_scalar();
        
        Ok(magnitude)
    }
    
    /// Build temporal context from observation history
    fn build_temporal_context(&self) -> LonebothResult<Tensor<B, 2>> {
        if self.observation_history.is_empty() {
            return Ok(Tensor::zeros([1, self.config.observation_dimension * self.config.temporal_context_length], &self.device));
        }
        
        // Concatenate recent observations into temporal context
        let mut context_tensors = Vec::new();
        for obs in &self.observation_history {
            context_tensors.push(obs.clone());
        }
        
        // Pad with zeros if we don't have enough history
        while context_tensors.len() < self.config.temporal_context_length {
            context_tensors.insert(0, Tensor::zeros([1, self.config.observation_dimension], &self.device));
        }
        
        let temporal_context = Tensor::cat(context_tensors, 1);
        Ok(temporal_context)
    }
    
    /// Compute structural adaptation based on environmental patterns
    pub async fn compute_structural_adaptation(&mut self, env_state: &EnvironmentState<B>) -> LonebothResult<Tensor<B, 2>> {
        debug!("Computing structural adaptation");
        
        // Analyze structural patterns in the environment
        let structural_patterns = self.analyze_structural_patterns(env_state)?;
        
        // Detect structural changes requiring adaptation
        let adaptation_signal = self.detect_structural_changes(&structural_patterns)?;
        
        if adaptation_signal.abs().mean().into_scalar() > self.config.adaptation_threshold {
            self.adaptation_count += 1;
            warn!("Structural adaptation required, event #{}", self.adaptation_count);
        }
        
        Ok(adaptation_signal)
    }
    
    /// Analyze structural patterns in environmental observations
    fn analyze_structural_patterns(&self, env_state: &EnvironmentState<B>) -> LonebothResult<Tensor<B, 2>> {
        // Analyze frequency domain characteristics
        let frequency_analysis = self.analyze_frequency_patterns(&env_state.observation)?;
        
        // Analyze spatial correlations
        let spatial_analysis = self.analyze_spatial_patterns(&env_state.observation)?;
        
        // Combine analyses
        let combined_patterns = frequency_analysis + spatial_analysis;
        Ok(combined_patterns)
    }
    
    /// Analyze frequency domain patterns (simplified implementation)
    fn analyze_frequency_patterns(&self, observation: &Tensor<B, 2>) -> LonebothResult<Tensor<B, 2>> {
        // Simplified frequency analysis using convolution-like operations
        // In practice, this would use FFT or other frequency domain methods
        let smoothed = observation.clone() * 0.9;
        Ok(smoothed)
    }
    
    /// Analyze spatial correlation patterns
    fn analyze_spatial_patterns(&self, observation: &Tensor<B, 2>) -> LonebothResult<Tensor<B, 2>> {
        // Compute local correlations (simplified)
        let shifted = observation.clone().roll(&[1], &[1]);
        let correlation = observation.clone() * shifted;
        Ok(correlation)
    }
    
    /// Detect structural changes requiring adaptation
    fn detect_structural_changes(&self, patterns: &Tensor<B, 2>) -> LonebothResult<Tensor<B, 2>> {
        // Compare current patterns with expected baseline
        let baseline = Tensor::zeros_like(patterns);
        let change_signal = patterns.clone() - baseline;
        
        // Apply threshold to detect significant changes
        let threshold = self.config.adaptation_threshold;
        let adaptation_mask = change_signal.clone().abs().greater_elem(threshold);
        let adaptation_signal = change_signal * adaptation_mask.float();
        
        Ok(adaptation_signal)
    }
    
    /// Get adaptation event count
    pub fn adaptation_count(&self) -> u64 {
        self.adaptation_count
    }
    
    /// Create observation from raw data
    pub fn create_observation(&self, data: Vec<f32>, metadata: Option<ObservationMetadata>) -> Observation {
        Observation {
            data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metadata,
        }
    }
    
    /// Convert observation to tensor
    pub fn observation_to_tensor(&self, observation: &Observation) -> LonebothResult<Tensor<B, 2>> {
        if observation.data.len() != self.config.observation_dimension {
            return Err(anyhow!(
                "Observation dimension mismatch: expected {}, got {}",
                self.config.observation_dimension,
                observation.data.len()
            ));
        }
        
        let tensor = Tensor::from_floats([observation.data.as_slice()], &self.device);
        Ok(tensor)
    }
}

impl<B: Backend> EnvironmentState<B> {
    /// Get current observation tensor
    pub fn current_observation(&self) -> Tensor<B, 2> {
        self.observation.clone()
    }
    
    /// Get temporal context tensor
    pub fn temporal_context(&self) -> Tensor<B, 2> {
        self.temporal_context.clone()
    }
    
    /// Get change magnitude
    pub fn change_magnitude(&self) -> f32 {
        self.change_magnitude
    }
    
    /// Get state timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }
    
    /// Check if significant change has occurred
    pub fn has_significant_change(&self, threshold: f32) -> bool {
        self.change_magnitude > threshold
    }
}

impl Default for ObservationMetadata {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            confidence: 1.0,
            context: Vec::new(),
        }
    }
}