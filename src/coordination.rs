//! Coordination system for individual and group algorithm execution
//! 
//! Manages coordination between multiple algorithms and consensus mechanisms.

use crate::algorithms::Algorithm;
use crate::Result;

/// Coordination mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinationMode {
    /// Individual algorithm execution
    Individual,
    /// Group coordination with consensus
    Group,
    /// Hybrid mode (individual + group verification)
    Hybrid,
}

/// Coordination system implementation
pub struct CoordinationSystem {
    mode: CoordinationMode,
    consensus_threshold: f32,
}

impl CoordinationSystem {
    /// Create a new coordination system
    pub fn new(mode: CoordinationMode) -> Self {
        Self {
            mode,
            consensus_threshold: 0.8, // 80% consensus required
        }
    }

    /// Process input through coordination system
    pub fn process(&self, algorithm: &dyn Algorithm, input: &[f32]) -> Result<Vec<f32>> {
        match self.mode {
            CoordinationMode::Individual => self.process_individual(algorithm, input),
            CoordinationMode::Group => self.process_group(algorithm, input),
            CoordinationMode::Hybrid => self.process_hybrid(algorithm, input),
        }
    }

    /// Process using individual algorithm
    fn process_individual(&self, algorithm: &dyn Algorithm, input: &[f32]) -> Result<Vec<f32>> {
        if !algorithm.is_ready() {
            return Err("Algorithm not ready".into());
        }
        
        algorithm.process(input)
    }

    /// Process using group coordination
    fn process_group(&self, algorithm: &dyn Algorithm, input: &[f32]) -> Result<Vec<f32>> {
        // For now, simulate group processing with multiple runs
        let mut results = Vec::new();
        
        // Run algorithm multiple times with slight variations
        for i in 0..3 {
            let mut modified_input = input.to_vec();
            
            // Add small variation to simulate different group members
            let variation = 0.01 * i as f32;
            for value in &mut modified_input {
                *value += variation;
            }
            
            let result = algorithm.process(&modified_input)?;
            results.push(result);
        }
        
        // Calculate consensus result
        self.calculate_consensus(&results)
    }

    /// Process using hybrid mode
    fn process_hybrid(&self, algorithm: &dyn Algorithm, input: &[f32]) -> Result<Vec<f32>> {
        // Individual processing
        let individual_result = self.process_individual(algorithm, input)?;
        
        // Group verification
        let group_result = self.process_group(algorithm, input)?;
        
        // Combine results with weighted average
        let mut combined_result = Vec::with_capacity(individual_result.len());
        
        for (&individual, &group) in individual_result.iter().zip(group_result.iter()) {
            let weight = 0.7; // 70% individual, 30% group
            let combined = individual * weight + group * (1.0 - weight);
            combined_result.push(combined);
        }
        
        Ok(combined_result)
    }

    /// Calculate consensus from multiple results
    fn calculate_consensus(&self, results: &[Vec<f32>]) -> Result<Vec<f32>> {
        if results.is_empty() {
            return Err("No results to calculate consensus".into());
        }
        
        let result_len = results[0].len();
        let mut consensus_result = vec![0.0; result_len];
        
        // Calculate weighted average
        for result in results {
            for (i, &value) in result.iter().enumerate() {
                consensus_result[i] += value;
            }
        }
        
        // Average the values
        let count = results.len() as f32;
        for value in &mut consensus_result {
            *value /= count;
        }
        
        Ok(consensus_result)
    }

    /// Get current coordination mode
    pub fn mode(&self) -> CoordinationMode {
        self.mode
    }

    /// Set consensus threshold
    pub fn set_consensus_threshold(&mut self, threshold: f32) {
        self.consensus_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get consensus threshold
    pub fn consensus_threshold(&self) -> f32 {
        self.consensus_threshold
    }
}

/// Consensus result with confidence metrics
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub result: Vec<f32>,
    pub confidence: f32,
    pub agreement_level: f32,
}

impl ConsensusResult {
    /// Create a new consensus result
    pub fn new(result: Vec<f32>, confidence: f32, agreement_level: f32) -> Self {
        Self {
            result,
            confidence,
            agreement_level,
        }
    }
    
    /// Check if consensus meets threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.agreement_level >= threshold
    }
}