//! Algorithm management and execution

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AlgorithmError {
    #[error("Algorithm not found: {name}")]
    NotFound { name: String },
    #[error("Algorithm execution failed: {reason}")]
    ExecutionFailed { reason: String },
    #[error("Invalid algorithm type: {algorithm_type}")]
    InvalidType { algorithm_type: String },
}

/// Algorithm execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmResult {
    pub data: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub execution_time_ms: u64,
}

/// Algorithm types supported by the framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmType {
    /// Static, compile-time algorithms
    Static,
    /// Dynamic, runtime-loaded algorithms
    Dynamic,
    /// Fuzzy/adaptive algorithms
    Fuzzy,
}

/// Core algorithm trait
#[async_trait]
pub trait Algorithm: Send + Sync {
    async fn execute(&self, input: &[f32]) -> Result<AlgorithmResult, AlgorithmError>;
    fn get_type(&self) -> AlgorithmType;
    fn get_name(&self) -> &str;
    fn get_version(&self) -> &str;
}

/// Static algorithm implementation
pub struct StaticAlgorithm {
    name: String,
    version: String,
    function: Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>,
}

impl StaticAlgorithm {
    pub fn new<F>(name: String, version: String, function: F) -> Self
    where
        F: Fn(&[f32]) -> Vec<f32> + Send + Sync + 'static,
    {
        Self {
            name,
            version,
            function: Box::new(function),
        }
    }
}

#[async_trait]
impl Algorithm for StaticAlgorithm {
    async fn execute(&self, input: &[f32]) -> Result<AlgorithmResult, AlgorithmError> {
        let start = std::time::Instant::now();
        let result = (self.function)(input);
        let execution_time = start.elapsed().as_millis() as u64;
        
        Ok(AlgorithmResult {
            data: result,
            metadata: HashMap::new(),
            execution_time_ms: execution_time,
        })
    }
    
    fn get_type(&self) -> AlgorithmType {
        AlgorithmType::Static
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_version(&self) -> &str {
        &self.version
    }
}

/// Dynamic algorithm implementation
pub struct DynamicAlgorithm {
    name: String,
    version: String,
    // In a real implementation, this would load from dylib
    placeholder_function: Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>,
}

impl DynamicAlgorithm {
    pub fn new(name: String, version: String) -> Self {
        // Placeholder implementation - in reality would load from dylib
        let placeholder_function = Box::new(|input: &[f32]| {
            // Simple transformation as placeholder
            input.iter().map(|x| x * 2.0).collect()
        });
        
        Self {
            name,
            version,
            placeholder_function,
        }
    }
}

#[async_trait]
impl Algorithm for DynamicAlgorithm {
    async fn execute(&self, input: &[f32]) -> Result<AlgorithmResult, AlgorithmError> {
        let start = std::time::Instant::now();
        let result = (self.placeholder_function)(input);
        let execution_time = start.elapsed().as_millis() as u64;
        
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "dynamic".to_string());
        
        Ok(AlgorithmResult {
            data: result,
            metadata,
            execution_time_ms: execution_time,
        })
    }
    
    fn get_type(&self) -> AlgorithmType {
        AlgorithmType::Dynamic
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_version(&self) -> &str {
        &self.version
    }
}

/// Fuzzy/adaptive algorithm implementation
pub struct FuzzyAlgorithm {
    name: String,
    version: String,
    adaptation_factor: f32,
    // Learning parameters
    history: Vec<AlgorithmResult>,
}

impl FuzzyAlgorithm {
    pub fn new(name: String, version: String, adaptation_factor: f32) -> Self {
        Self {
            name,
            version,
            adaptation_factor,
            history: Vec::new(),
        }
    }
    
    fn adapt_parameters(&mut self, input: &[f32]) -> f32 {
        // Simple adaptation based on input variance
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let variance = input.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / input.len() as f32;
        
        self.adaptation_factor * (1.0 + variance.sqrt())
    }
}

#[async_trait]
impl Algorithm for FuzzyAlgorithm {
    async fn execute(&self, input: &[f32]) -> Result<AlgorithmResult, AlgorithmError> {
        let start = std::time::Instant::now();
        
        // Adaptive processing
        let mut algorithm = self.clone();
        let adapted_factor = algorithm.adapt_parameters(input);
        
        // Apply fuzzy transformation
        let result: Vec<f32> = input.iter()
            .map(|x| x * adapted_factor)
            .collect();
        
        let execution_time = start.elapsed().as_millis() as u64;
        
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "fuzzy".to_string());
        metadata.insert("adaptation_factor".to_string(), adapted_factor.to_string());
        
        let algorithm_result = AlgorithmResult {
            data: result,
            metadata,
            execution_time_ms: execution_time,
        };
        
        Ok(algorithm_result)
    }
    
    fn get_type(&self) -> AlgorithmType {
        AlgorithmType::Fuzzy
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_version(&self) -> &str {
        &self.version
    }
}

impl Clone for FuzzyAlgorithm {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            version: self.version.clone(),
            adaptation_factor: self.adaptation_factor,
            history: self.history.clone(),
        }
    }
}

/// Algorithm registry for managing different algorithm implementations
pub struct AlgorithmRegistry {
    algorithms: HashMap<String, Box<dyn Algorithm>>,
}

impl AlgorithmRegistry {
    pub fn new() -> Self {
        Self {
            algorithms: HashMap::new(),
        }
    }
    
    pub fn register(&mut self, algorithm: Box<dyn Algorithm>) {
        let name = algorithm.get_name().to_string();
        self.algorithms.insert(name, algorithm);
    }
    
    pub fn get(&self, name: &str) -> Option<&dyn Algorithm> {
        self.algorithms.get(name).map(|a| a.as_ref())
    }
    
    pub fn list_algorithms(&self) -> Vec<&str> {
        self.algorithms.keys().map(|k| k.as_str()).collect()
    }
}

impl Default for AlgorithmRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        
        // Register built-in algorithms
        registry.register(Box::new(StaticAlgorithm::new(
            "simple_add".to_string(),
            "1.0.0".to_string(),
            |input| input.iter().map(|x| x + 1.0).collect(),
        )));
        
        registry.register(Box::new(DynamicAlgorithm::new(
            "dynamic_multiply".to_string(),
            "1.0.0".to_string(),
        )));
        
        registry.register(Box::new(FuzzyAlgorithm::new(
            "fuzzy_adaptive".to_string(),
            "1.0.0".to_string(),
            0.5,
        )));
        
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_static_algorithm() {
        let algorithm = StaticAlgorithm::new(
            "test".to_string(),
            "1.0.0".to_string(),
            |input| input.iter().map(|x| x * 2.0).collect(),
        );
        
        let input = vec![1.0, 2.0, 3.0];
        let result = algorithm.execute(&input).await.unwrap();
        
        assert_eq!(result.data, vec![2.0, 4.0, 6.0]);
        assert!(result.execution_time_ms > 0);
    }

    #[tokio::test]
    async fn test_dynamic_algorithm() {
        let algorithm = DynamicAlgorithm::new(
            "test_dynamic".to_string(),
            "1.0.0".to_string(),
        );
        
        let input = vec![1.0, 2.0, 3.0];
        let result = algorithm.execute(&input).await.unwrap();
        
        assert_eq!(result.data, vec![2.0, 4.0, 6.0]);
        assert_eq!(result.metadata.get("type").unwrap(), "dynamic");
    }

    #[tokio::test]
    async fn test_fuzzy_algorithm() {
        let algorithm = FuzzyAlgorithm::new(
            "test_fuzzy".to_string(),
            "1.0.0".to_string(),
            0.5,
        );
        
        let input = vec![1.0, 2.0, 3.0];
        let result = algorithm.execute(&input).await.unwrap();
        
        assert!(result.data.len() == input.len());
        assert_eq!(result.metadata.get("type").unwrap(), "fuzzy");
        assert!(result.metadata.contains_key("adaptation_factor"));
    }

    #[test]
    fn test_algorithm_registry() {
        let registry = AlgorithmRegistry::default();
        let algorithms = registry.list_algorithms();
        
        assert!(algorithms.contains(&"simple_add"));
        assert!(algorithms.contains(&"dynamic_multiply"));
        assert!(algorithms.contains(&"fuzzy_adaptive"));
        
        let algorithm = registry.get("simple_add");
        assert!(algorithm.is_some());
    }
}