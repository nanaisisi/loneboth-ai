//! Algorithm coordination system for individual and group execution

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

use crate::algorithm::{Algorithm, AlgorithmResult};

#[derive(Error, Debug)]
pub enum CoordinationError {
    #[error("Algorithm not found: {name}")]
    AlgorithmNotFound { name: String },
    #[error("Coordination failed: {reason}")]
    CoordinationFailed { reason: String },
    #[error("Consensus not reached: threshold {threshold}, actual {actual}")]
    ConsensusNotReached { threshold: f32, actual: f32 },
    #[error("Timeout reached: {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
}

/// Coordination result containing aggregated algorithm results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationResult {
    pub results: HashMap<String, AlgorithmResult>,
    pub consensus_score: f32,
    pub execution_time_ms: u64,
    pub coordination_type: CoordinationType,
}

/// Types of coordination supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationType {
    /// Individual algorithm execution
    Individual,
    /// Group consensus-based execution
    GroupConsensus,
    /// Parallel group execution
    GroupParallel,
}

/// Core coordination trait
#[async_trait]
pub trait Coordinator: Send + Sync {
    async fn execute(&self, input: &[f32]) -> Result<CoordinationResult, CoordinationError>;
    fn get_coordination_type(&self) -> CoordinationType;
    fn add_algorithm(&mut self, name: String, algorithm: Box<dyn Algorithm>);
    fn remove_algorithm(&mut self, name: &str) -> Option<Box<dyn Algorithm>>;
}

/// Individual algorithm coordinator (単体協調)
/// Executes algorithms sequentially with minimal overhead
pub struct IndividualCoordinator {
    algorithms: HashMap<String, Box<dyn Algorithm>>,
    current_algorithm: Option<String>,
}

impl IndividualCoordinator {
    pub fn new() -> Self {
        Self {
            algorithms: HashMap::new(),
            current_algorithm: None,
        }
    }
    
    pub fn set_current_algorithm(&mut self, name: String) -> Result<(), CoordinationError> {
        if self.algorithms.contains_key(&name) {
            self.current_algorithm = Some(name);
            Ok(())
        } else {
            Err(CoordinationError::AlgorithmNotFound { name })
        }
    }
    
    pub async fn execute_algorithm(&self, name: &str, input: &[f32]) -> Result<AlgorithmResult, CoordinationError> {
        match self.algorithms.get(name) {
            Some(algorithm) => {
                algorithm.execute(input).await
                    .map_err(|e| CoordinationError::CoordinationFailed { 
                        reason: format!("Algorithm execution failed: {}", e) 
                    })
            }
            None => Err(CoordinationError::AlgorithmNotFound { name: name.to_string() })
        }
    }
}

#[async_trait]
impl Coordinator for IndividualCoordinator {
    async fn execute(&self, input: &[f32]) -> Result<CoordinationResult, CoordinationError> {
        let start = std::time::Instant::now();
        
        let algorithm_name = self.current_algorithm.as_ref()
            .ok_or_else(|| CoordinationError::CoordinationFailed { 
                reason: "No algorithm selected".to_string() 
            })?;
        
        let result = self.execute_algorithm(algorithm_name, input).await?;
        let execution_time = start.elapsed().as_millis() as u64;
        
        let mut results = HashMap::new();
        results.insert(algorithm_name.clone(), result);
        
        Ok(CoordinationResult {
            results,
            consensus_score: 1.0, // Individual execution always has perfect "consensus"
            execution_time_ms: execution_time,
            coordination_type: CoordinationType::Individual,
        })
    }
    
    fn get_coordination_type(&self) -> CoordinationType {
        CoordinationType::Individual
    }
    
    fn add_algorithm(&mut self, name: String, algorithm: Box<dyn Algorithm>) {
        self.algorithms.insert(name, algorithm);
    }
    
    fn remove_algorithm(&mut self, name: &str) -> Option<Box<dyn Algorithm>> {
        self.algorithms.remove(name)
    }
}

/// Group algorithm coordinator (群体協調)
/// Supports multi-agent systems with consensus mechanisms
pub struct GroupCoordinator {
    algorithms: Arc<RwLock<HashMap<String, Box<dyn Algorithm>>>>,
    consensus_threshold: f32,
    timeout_ms: u64,
}

impl GroupCoordinator {
    pub fn new() -> Self {
        Self {
            algorithms: Arc::new(RwLock::new(HashMap::new())),
            consensus_threshold: 0.8,
            timeout_ms: 5000,
        }
    }
    
    pub fn with_consensus_threshold(mut self, threshold: f32) -> Self {
        self.consensus_threshold = threshold.clamp(0.0, 1.0);
        self
    }
    
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }
    
    /// Execute algorithms in parallel and compute consensus
    pub async fn execute_parallel(&self, input: &[f32]) -> Result<CoordinationResult, CoordinationError> {
        let start = std::time::Instant::now();
        let algorithms = self.algorithms.read().await;
        
        if algorithms.is_empty() {
            return Err(CoordinationError::CoordinationFailed { 
                reason: "No algorithms available".to_string() 
            });
        }
        
        // Execute all algorithms in parallel
        let mut tasks = Vec::new();
        for (name, _algorithm) in algorithms.iter() {
            let name = name.clone();
            let input = input.to_vec();
            
            // Create a task for each algorithm execution
            // Note: In a real implementation, we'd need to make Algorithm Clone or use Arc
            // For now, we'll execute sequentially but structure for parallel
            tasks.push((name, input));
        }
        
        // Execute tasks (simplified sequential execution for prototype)
        let mut results = HashMap::new();
        for (name, input) in tasks {
            if let Some(algorithm) = algorithms.get(&name) {
                match algorithm.execute(&input).await {
                    Ok(result) => {
                        results.insert(name, result);
                    }
                    Err(e) => {
                        log::warn!("Algorithm {} failed: {}", name, e);
                    }
                }
            }
        }
        
        let execution_time = start.elapsed().as_millis() as u64;
        
        // Compute consensus score
        let consensus_score = self.compute_consensus(&results).await;
        
        // Check if consensus threshold is met
        if consensus_score < self.consensus_threshold {
            return Err(CoordinationError::ConsensusNotReached { 
                threshold: self.consensus_threshold, 
                actual: consensus_score 
            });
        }
        
        Ok(CoordinationResult {
            results,
            consensus_score,
            execution_time_ms: execution_time,
            coordination_type: CoordinationType::GroupParallel,
        })
    }
    
    /// Execute algorithms and wait for consensus
    pub async fn execute_consensus(&self, input: &[f32]) -> Result<CoordinationResult, CoordinationError> {
        let start = std::time::Instant::now();
        let algorithms = self.algorithms.read().await;
        
        if algorithms.is_empty() {
            return Err(CoordinationError::CoordinationFailed { 
                reason: "No algorithms available".to_string() 
            });
        }
        
        let mut results = HashMap::new();
        let mut consensus_rounds = 0;
        let max_rounds = 3;
        
        // Consensus-based execution with multiple rounds
        while consensus_rounds < max_rounds {
            results.clear();
            
            // Execute all algorithms
            for (name, algorithm) in algorithms.iter() {
                match algorithm.execute(input).await {
                    Ok(result) => {
                        results.insert(name.clone(), result);
                    }
                    Err(e) => {
                        log::warn!("Algorithm {} failed in round {}: {}", name, consensus_rounds, e);
                    }
                }
            }
            
            // Check consensus
            let consensus_score = self.compute_consensus(&results).await;
            if consensus_score >= self.consensus_threshold {
                let execution_time = start.elapsed().as_millis() as u64;
                return Ok(CoordinationResult {
                    results,
                    consensus_score,
                    execution_time_ms: execution_time,
                    coordination_type: CoordinationType::GroupConsensus,
                });
            }
            
            consensus_rounds += 1;
            
            // Check timeout
            if start.elapsed().as_millis() as u64 > self.timeout_ms {
                return Err(CoordinationError::Timeout { timeout_ms: self.timeout_ms });
            }
        }
        
        Err(CoordinationError::ConsensusNotReached { 
            threshold: self.consensus_threshold, 
            actual: self.compute_consensus(&results).await 
        })
    }
    
    /// Compute consensus score based on result similarity
    async fn compute_consensus(&self, results: &HashMap<String, AlgorithmResult>) -> f32 {
        if results.len() < 2 {
            return 1.0; // Single result always has perfect consensus
        }
        
        let result_vectors: Vec<&Vec<f32>> = results.values().map(|r| &r.data).collect();
        if result_vectors.is_empty() {
            return 0.0;
        }
        
        // Compute pairwise similarities and average them
        let mut total_similarity = 0.0;
        let mut comparisons = 0;
        
        for i in 0..result_vectors.len() {
            for j in (i + 1)..result_vectors.len() {
                let similarity = self.compute_similarity(result_vectors[i], result_vectors[j]);
                total_similarity += similarity;
                comparisons += 1;
            }
        }
        
        if comparisons > 0 {
            total_similarity / comparisons as f32
        } else {
            1.0
        }
    }
    
    /// Compute similarity between two result vectors using cosine similarity
    fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }
}

#[async_trait]
impl Coordinator for GroupCoordinator {
    async fn execute(&self, input: &[f32]) -> Result<CoordinationResult, CoordinationError> {
        self.execute_consensus(input).await
    }
    
    fn get_coordination_type(&self) -> CoordinationType {
        CoordinationType::GroupConsensus
    }
    
    fn add_algorithm(&mut self, name: String, algorithm: Box<dyn Algorithm>) {
        // Note: This is simplified for the prototype. In a real async implementation,
        // we'd need to handle the RwLock properly
        tokio::task::block_in_place(|| {
            let mut algorithms = futures::executor::block_on(self.algorithms.write());
            algorithms.insert(name, algorithm);
        });
    }
    
    fn remove_algorithm(&mut self, name: &str) -> Option<Box<dyn Algorithm>> {
        tokio::task::block_in_place(|| {
            let mut algorithms = futures::executor::block_on(self.algorithms.write());
            algorithms.remove(name)
        })
    }
}

/// Coordination manager for handling different coordination strategies
pub struct CoordinationManager {
    coordinators: HashMap<String, Box<dyn Coordinator>>,
    active_coordinator: Option<String>,
}

impl CoordinationManager {
    pub fn new() -> Self {
        Self {
            coordinators: HashMap::new(),
            active_coordinator: None,
        }
    }
    
    pub fn add_coordinator(&mut self, name: String, coordinator: Box<dyn Coordinator>) {
        self.coordinators.insert(name, coordinator);
    }
    
    pub fn set_active_coordinator(&mut self, name: String) -> Result<(), CoordinationError> {
        if self.coordinators.contains_key(&name) {
            self.active_coordinator = Some(name);
            Ok(())
        } else {
            Err(CoordinationError::CoordinationFailed { 
                reason: format!("Coordinator {} not found", name) 
            })
        }
    }
    
    pub async fn execute(&self, input: &[f32]) -> Result<CoordinationResult, CoordinationError> {
        let coordinator_name = self.active_coordinator.as_ref()
            .ok_or_else(|| CoordinationError::CoordinationFailed { 
                reason: "No active coordinator".to_string() 
            })?;
        
        match self.coordinators.get(coordinator_name) {
            Some(coordinator) => coordinator.execute(input).await,
            None => Err(CoordinationError::CoordinationFailed { 
                reason: format!("Active coordinator {} not found", coordinator_name) 
            })
        }
    }
}

impl Default for CoordinationManager {
    fn default() -> Self {
        let mut manager = Self::new();
        
        // Add default coordinators
        manager.add_coordinator("individual".to_string(), Box::new(IndividualCoordinator::new()));
        manager.add_coordinator("group".to_string(), Box::new(GroupCoordinator::new()));
        
        manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::StaticAlgorithm;
    use tokio_test;

    #[tokio::test]
    async fn test_individual_coordinator() {
        let mut coordinator = IndividualCoordinator::new();
        
        let algorithm = StaticAlgorithm::new(
            "test".to_string(),
            "1.0.0".to_string(),
            |input| input.iter().map(|x| x * 2.0).collect(),
        );
        
        coordinator.add_algorithm("test".to_string(), Box::new(algorithm));
        coordinator.set_current_algorithm("test".to_string()).unwrap();
        
        let input = vec![1.0, 2.0, 3.0];
        let result = coordinator.execute(&input).await.unwrap();
        
        assert_eq!(result.consensus_score, 1.0);
        assert!(matches!(result.coordination_type, CoordinationType::Individual));
        assert!(result.results.contains_key("test"));
    }

    #[tokio::test]
    async fn test_group_coordinator() {
        let mut coordinator = GroupCoordinator::new()
            .with_consensus_threshold(0.5)
            .with_timeout(1000);
        
        let algorithm1 = StaticAlgorithm::new(
            "algo1".to_string(),
            "1.0.0".to_string(),
            |input| input.iter().map(|x| x * 2.0).collect(),
        );
        
        let algorithm2 = StaticAlgorithm::new(
            "algo2".to_string(),
            "1.0.0".to_string(),
            |input| input.iter().map(|x| x * 2.1).collect(), // Slightly different for testing
        );
        
        coordinator.add_algorithm("algo1".to_string(), Box::new(algorithm1));
        coordinator.add_algorithm("algo2".to_string(), Box::new(algorithm2));
        
        let input = vec![1.0, 2.0, 3.0];
        let result = coordinator.execute(&input).await.unwrap();
        
        assert!(result.consensus_score > 0.5);
        assert!(matches!(result.coordination_type, CoordinationType::GroupConsensus));
        assert_eq!(result.results.len(), 2);
    }

    #[test]
    fn test_coordination_manager() {
        let manager = CoordinationManager::default();
        
        // Test that default coordinators are available
        assert!(manager.coordinators.contains_key("individual"));
        assert!(manager.coordinators.contains_key("group"));
    }

    #[test]
    fn test_similarity_computation() {
        let coordinator = GroupCoordinator::new();
        
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = vec![2.0, 4.0, 6.0];
        
        // Identical vectors should have similarity 1.0
        assert!((coordinator.compute_similarity(&a, &b) - 1.0).abs() < 0.001);
        
        // Proportional vectors should have high similarity
        assert!(coordinator.compute_similarity(&a, &c) > 0.99);
        
        // Different length vectors should have similarity 0.0
        let d = vec![1.0, 2.0];
        assert_eq!(coordinator.compute_similarity(&a, &d), 0.0);
    }
}