//! Consistency verification system for algorithm integrity and data validation

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::algorithm::AlgorithmResult;

#[derive(Error, Debug)]
pub enum VerificationError {
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },
    #[error("Consistency check failed: {reason}")]
    ConsistencyFailed { reason: String },
    #[error("Performance regression detected: current {current}ms > threshold {threshold}ms")]
    PerformanceRegression { current: u64, threshold: u64 },
    #[error("Data integrity violation: {field}")]
    DataIntegrityViolation { field: String },
}

/// Verification result containing detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub passed: bool,
    pub checks_performed: Vec<String>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub performance_metrics: PerformanceMetrics,
    pub integrity_score: f32,
}

/// Performance metrics for algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub execution_time_ms: u64,
    pub memory_usage_mb: Option<u64>,
    pub cpu_usage_percent: Option<f32>,
    pub gpu_usage_percent: Option<f32>,
    pub throughput_ops_per_sec: Option<f32>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time_ms: 0,
            memory_usage_mb: None,
            cpu_usage_percent: None,
            gpu_usage_percent: None,
            throughput_ops_per_sec: None,
        }
    }
}

/// Core verification trait
#[async_trait]
pub trait Verifier: Send + Sync {
    async fn verify(&self, input: &[f32], output: &AlgorithmResult) -> Result<VerificationResult, VerificationError>;
    fn get_name(&self) -> &str;
    fn get_description(&self) -> &str;
}

/// Algorithm integrity verifier
pub struct AlgorithmIntegrityVerifier {
    name: String,
    expected_checksums: HashMap<String, String>,
    performance_thresholds: HashMap<String, u64>,
}

impl AlgorithmIntegrityVerifier {
    pub fn new() -> Self {
        Self {
            name: "AlgorithmIntegrityVerifier".to_string(),
            expected_checksums: HashMap::new(),
            performance_thresholds: HashMap::new(),
        }
    }
    
    pub fn add_checksum(&mut self, algorithm_name: String, checksum: String) {
        self.expected_checksums.insert(algorithm_name, checksum);
    }
    
    pub fn add_performance_threshold(&mut self, algorithm_name: String, threshold_ms: u64) {
        self.performance_thresholds.insert(algorithm_name, threshold_ms);
    }
    
    /// Compute checksum for algorithm result
    fn compute_result_checksum(&self, result: &AlgorithmResult) -> String {
        let mut hasher = DefaultHasher::new();
        
        // Hash the result data
        for value in &result.data {
            value.to_bits().hash(&mut hasher);
        }
        
        // Hash metadata
        for (key, value) in &result.metadata {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
        
        format!("{:016x}", hasher.finish())
    }
    
    /// Check performance regression
    fn check_performance(&self, algorithm_name: &str, execution_time: u64) -> Result<(), VerificationError> {
        if let Some(&threshold) = self.performance_thresholds.get(algorithm_name) {
            if execution_time > threshold {
                return Err(VerificationError::PerformanceRegression {
                    current: execution_time,
                    threshold,
                });
            }
        }
        Ok(())
    }
    
    /// Validate result consistency
    fn validate_consistency(&self, input: &[f32], result: &AlgorithmResult) -> Result<Vec<String>, Vec<String>> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        
        // Check if result data is empty
        if result.data.is_empty() {
            errors.push("Result data is empty".to_string());
        }
        
        // Check for NaN or infinite values
        for (i, value) in result.data.iter().enumerate() {
            if value.is_nan() {
                errors.push(format!("NaN value found at index {}", i));
            } else if value.is_infinite() {
                warnings.push(format!("Infinite value found at index {}", i));
            }
        }
        
        // Check if result size is reasonable compared to input
        let size_ratio = result.data.len() as f32 / input.len() as f32;
        if size_ratio > 100.0 {
            warnings.push(format!("Result size is {}x larger than input", size_ratio as u32));
        } else if size_ratio < 0.01 {
            warnings.push(format!("Result size is {}x smaller than input", (1.0 / size_ratio) as u32));
        }
        
        // Check execution time reasonableness
        if result.execution_time_ms > 60000 { // More than 1 minute
            warnings.push(format!("Execution time is very high: {}ms", result.execution_time_ms));
        }
        
        Ok(warnings)
    }
}

#[async_trait]
impl Verifier for AlgorithmIntegrityVerifier {
    async fn verify(&self, input: &[f32], output: &AlgorithmResult) -> Result<VerificationResult, VerificationError> {
        let mut checks_performed = Vec::new();
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut passed = true;
        
        // Perform consistency validation
        checks_performed.push("consistency_validation".to_string());
        match self.validate_consistency(input, output) {
            Ok(mut w) => warnings.append(&mut w),
            Err(mut e) => {
                errors.append(&mut e);
                passed = false;
            }
        }
        
        // Check checksum if available
        if let Some(algorithm_type) = output.metadata.get("type") {
            checks_performed.push("checksum_verification".to_string());
            let actual_checksum = self.compute_result_checksum(output);
            
            if let Some(expected) = self.expected_checksums.get(algorithm_type) {
                if &actual_checksum != expected {
                    errors.push(format!("Checksum mismatch for {}: expected {}, got {}", 
                                      algorithm_type, expected, actual_checksum));
                    passed = false;
                }
            } else {
                warnings.push(format!("No expected checksum for algorithm type: {}", algorithm_type));
            }
        }
        
        // Performance regression check
        if let Some(algorithm_type) = output.metadata.get("type") {
            checks_performed.push("performance_check".to_string());
            if let Err(e) = self.check_performance(algorithm_type, output.execution_time_ms) {
                errors.push(e.to_string());
                passed = false;
            }
        }
        
        // Calculate integrity score
        let total_checks = checks_performed.len() as f32;
        let failed_checks = errors.len() as f32;
        let warning_weight = 0.1;
        let warning_penalty = warnings.len() as f32 * warning_weight;
        
        let integrity_score = if total_checks > 0.0 {
            ((total_checks - failed_checks - warning_penalty) / total_checks).clamp(0.0, 1.0)
        } else {
            1.0
        };
        
        Ok(VerificationResult {
            passed: passed && integrity_score > 0.8,
            checks_performed,
            warnings,
            errors,
            performance_metrics: PerformanceMetrics {
                execution_time_ms: output.execution_time_ms,
                ..Default::default()
            },
            integrity_score,
        })
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_description(&self) -> &str {
        "Verifies algorithm integrity through checksum validation and performance monitoring"
    }
}

/// Data integrity verifier for input/output validation
pub struct DataIntegrityVerifier {
    name: String,
    input_constraints: Vec<DataConstraint>,
    output_constraints: Vec<DataConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConstraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub parameters: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    Range { min: f32, max: f32 },
    Size { min_size: usize, max_size: usize },
    Pattern { regex: String },
    Custom { validator_name: String },
}

impl DataIntegrityVerifier {
    pub fn new() -> Self {
        Self {
            name: "DataIntegrityVerifier".to_string(),
            input_constraints: Vec::new(),
            output_constraints: Vec::new(),
        }
    }
    
    pub fn add_input_constraint(&mut self, constraint: DataConstraint) {
        self.input_constraints.push(constraint);
    }
    
    pub fn add_output_constraint(&mut self, constraint: DataConstraint) {
        self.output_constraints.push(constraint);
    }
    
    /// Validate data against constraints
    fn validate_constraints(&self, data: &[f32], constraints: &[DataConstraint]) -> Result<Vec<String>, Vec<String>> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        
        for constraint in constraints {
            match &constraint.constraint_type {
                ConstraintType::Range { min, max } => {
                    for (i, value) in data.iter().enumerate() {
                        if *value < *min || *value > *max {
                            errors.push(format!("Value {} at index {} violates range constraint [{}, {}]", 
                                               value, i, min, max));
                        }
                    }
                }
                ConstraintType::Size { min_size, max_size } => {
                    if data.len() < *min_size {
                        errors.push(format!("Data size {} is below minimum {}", data.len(), min_size));
                    } else if data.len() > *max_size {
                        errors.push(format!("Data size {} exceeds maximum {}", data.len(), max_size));
                    }
                }
                ConstraintType::Pattern { regex: _ } => {
                    warnings.push("Pattern constraints not implemented for numeric data".to_string());
                }
                ConstraintType::Custom { validator_name } => {
                    warnings.push(format!("Custom validator '{}' not implemented", validator_name));
                }
            }
        }
        
        Ok(warnings)
    }
    
    /// Check data distribution properties
    fn analyze_distribution(&self, data: &[f32]) -> HashMap<String, f32> {
        if data.is_empty() {
            return HashMap::new();
        }
        
        let mut properties = HashMap::new();
        
        // Calculate mean
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        properties.insert("mean".to_string(), mean);
        
        // Calculate variance
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        properties.insert("variance".to_string(), variance);
        properties.insert("std_dev".to_string(), variance.sqrt());
        
        // Calculate min/max
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        properties.insert("min".to_string(), min);
        properties.insert("max".to_string(), max);
        properties.insert("range".to_string(), max - min);
        
        // Calculate median (approximate)
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted_data.len() % 2 == 0 {
            (sorted_data[sorted_data.len() / 2 - 1] + sorted_data[sorted_data.len() / 2]) / 2.0
        } else {
            sorted_data[sorted_data.len() / 2]
        };
        properties.insert("median".to_string(), median);
        
        properties
    }
}

#[async_trait]
impl Verifier for DataIntegrityVerifier {
    async fn verify(&self, input: &[f32], output: &AlgorithmResult) -> Result<VerificationResult, VerificationError> {
        let mut checks_performed = Vec::new();
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut passed = true;
        
        // Validate input constraints
        checks_performed.push("input_validation".to_string());
        match self.validate_constraints(input, &self.input_constraints) {
            Ok(mut w) => warnings.append(&mut w),
            Err(mut e) => {
                errors.append(&mut e);
                passed = false;
            }
        }
        
        // Validate output constraints
        checks_performed.push("output_validation".to_string());
        match self.validate_constraints(&output.data, &self.output_constraints) {
            Ok(mut w) => warnings.append(&mut w),
            Err(mut e) => {
                errors.append(&mut e);
                passed = false;
            }
        }
        
        // Analyze data distributions
        checks_performed.push("distribution_analysis".to_string());
        let input_properties = self.analyze_distribution(input);
        let output_properties = self.analyze_distribution(&output.data);
        
        // Check for suspicious distribution changes
        if let (Some(input_std), Some(output_std)) = 
            (input_properties.get("std_dev"), output_properties.get("std_dev")) {
            let std_ratio = output_std / input_std;
            if std_ratio > 10.0 {
                warnings.push(format!("Output standard deviation is {}x larger than input", std_ratio));
            } else if std_ratio < 0.1 {
                warnings.push(format!("Output standard deviation is {}x smaller than input", 1.0 / std_ratio));
            }
        }
        
        // Calculate integrity score
        let total_checks = checks_performed.len() as f32;
        let failed_checks = errors.len() as f32;
        let warning_weight = 0.2;
        let warning_penalty = warnings.len() as f32 * warning_weight;
        
        let integrity_score = if total_checks > 0.0 {
            ((total_checks - failed_checks - warning_penalty) / total_checks).clamp(0.0, 1.0)
        } else {
            1.0
        };
        
        Ok(VerificationResult {
            passed: passed && integrity_score > 0.7,
            checks_performed,
            warnings,
            errors,
            performance_metrics: PerformanceMetrics {
                execution_time_ms: output.execution_time_ms,
                ..Default::default()
            },
            integrity_score,
        })
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_description(&self) -> &str {
        "Validates data integrity through constraint checking and distribution analysis"
    }
}

/// Comprehensive verification system
pub struct VerificationSystem {
    verifiers: Vec<Box<dyn Verifier>>,
    verification_history: HashMap<String, Vec<VerificationResult>>,
    enabled: bool,
}

impl VerificationSystem {
    pub fn new() -> Self {
        Self {
            verifiers: Vec::new(),
            verification_history: HashMap::new(),
            enabled: true,
        }
    }
    
    pub fn add_verifier(&mut self, verifier: Box<dyn Verifier>) {
        self.verifiers.push(verifier);
    }
    
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Run all verifiers on the given input/output pair
    pub async fn verify_all(&mut self, algorithm_name: &str, input: &[f32], output: &AlgorithmResult) -> Result<VerificationResult, VerificationError> {
        if !self.enabled {
            return Ok(VerificationResult {
                passed: true,
                checks_performed: vec!["verification_disabled".to_string()],
                warnings: Vec::new(),
                errors: Vec::new(),
                performance_metrics: PerformanceMetrics {
                    execution_time_ms: output.execution_time_ms,
                    ..Default::default()
                },
                integrity_score: 1.0,
            });
        }
        
        let mut all_checks = Vec::new();
        let mut all_warnings = Vec::new();
        let mut all_errors = Vec::new();
        let mut total_score = 0.0;
        let mut all_passed = true;
        
        // Run all verifiers
        for verifier in &self.verifiers {
            log::debug!("Running verifier: {}", verifier.get_name());
            
            match verifier.verify(input, output).await {
                Ok(result) => {
                    all_checks.extend(result.checks_performed);
                    all_warnings.extend(result.warnings);
                    all_errors.extend(result.errors);
                    total_score += result.integrity_score;
                    all_passed = all_passed && result.passed;
                }
                Err(e) => {
                    all_errors.push(format!("Verifier {} failed: {}", verifier.get_name(), e));
                    all_passed = false;
                }
            }
        }
        
        // Calculate overall integrity score
        let overall_score = if !self.verifiers.is_empty() {
            total_score / self.verifiers.len() as f32
        } else {
            1.0
        };
        
        let final_result = VerificationResult {
            passed: all_passed && overall_score > 0.8,
            checks_performed: all_checks,
            warnings: all_warnings,
            errors: all_errors,
            performance_metrics: PerformanceMetrics {
                execution_time_ms: output.execution_time_ms,
                ..Default::default()
            },
            integrity_score: overall_score,
        };
        
        // Store in history
        self.verification_history
            .entry(algorithm_name.to_string())
            .or_insert_with(Vec::new)
            .push(final_result.clone());
        
        // Keep only last 100 results
        if let Some(history) = self.verification_history.get_mut(algorithm_name) {
            if history.len() > 100 {
                history.remove(0);
            }
        }
        
        Ok(final_result)
    }
    
    /// Get verification history for an algorithm
    pub fn get_history(&self, algorithm_name: &str) -> Option<&Vec<VerificationResult>> {
        self.verification_history.get(algorithm_name)
    }
    
    /// Get statistics for algorithm verification
    pub fn get_verification_stats(&self, algorithm_name: &str) -> Option<VerificationStats> {
        let history = self.verification_history.get(algorithm_name)?;
        
        if history.is_empty() {
            return None;
        }
        
        let total_verifications = history.len();
        let passed_verifications = history.iter().filter(|r| r.passed).count();
        let average_score = history.iter().map(|r| r.integrity_score).sum::<f32>() / total_verifications as f32;
        let latest_score = history.last().map(|r| r.integrity_score);
        
        Some(VerificationStats {
            total_verifications,
            passed_verifications,
            pass_rate: passed_verifications as f32 / total_verifications as f32,
            average_integrity_score: average_score,
            latest_integrity_score: latest_score,
        })
    }
    
    /// Clear verification history
    pub fn clear_history(&mut self, algorithm_name: Option<&str>) {
        match algorithm_name {
            Some(name) => {
                self.verification_history.remove(name);
            }
            None => {
                self.verification_history.clear();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerificationStats {
    pub total_verifications: usize,
    pub passed_verifications: usize,
    pub pass_rate: f32,
    pub average_integrity_score: f32,
    pub latest_integrity_score: Option<f32>,
}

impl Default for VerificationSystem {
    fn default() -> Self {
        let mut system = Self::new();
        
        // Add default verifiers
        system.add_verifier(Box::new(AlgorithmIntegrityVerifier::new()));
        system.add_verifier(Box::new(DataIntegrityVerifier::new()));
        
        system
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::AlgorithmResult;
    use tokio_test;

    #[tokio::test]
    async fn test_algorithm_integrity_verifier() {
        let verifier = AlgorithmIntegrityVerifier::new();
        
        let input = vec![1.0, 2.0, 3.0];
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "test".to_string());
        
        let output = AlgorithmResult {
            data: vec![2.0, 4.0, 6.0],
            metadata,
            execution_time_ms: 10,
        };
        
        let result = verifier.verify(&input, &output).await.unwrap();
        assert!(result.passed);
        assert!(!result.checks_performed.is_empty());
    }

    #[tokio::test]
    async fn test_data_integrity_verifier() {
        let mut verifier = DataIntegrityVerifier::new();
        
        // Add a range constraint
        verifier.add_input_constraint(DataConstraint {
            name: "input_range".to_string(),
            constraint_type: ConstraintType::Range { min: 0.0, max: 10.0 },
            parameters: HashMap::new(),
        });
        
        let input = vec![1.0, 2.0, 3.0];
        let output = AlgorithmResult {
            data: vec![2.0, 4.0, 6.0],
            metadata: HashMap::new(),
            execution_time_ms: 10,
        };
        
        let result = verifier.verify(&input, &output).await.unwrap();
        assert!(result.passed);
    }

    #[tokio::test]
    async fn test_data_integrity_verifier_violation() {
        let mut verifier = DataIntegrityVerifier::new();
        
        // Add a strict range constraint
        verifier.add_input_constraint(DataConstraint {
            name: "input_range".to_string(),
            constraint_type: ConstraintType::Range { min: 0.0, max: 1.0 },
            parameters: HashMap::new(),
        });
        
        let input = vec![5.0, 10.0, 15.0]; // Violates range constraint
        let output = AlgorithmResult {
            data: vec![10.0, 20.0, 30.0],
            metadata: HashMap::new(),
            execution_time_ms: 10,
        };
        
        let result = verifier.verify(&input, &output).await.unwrap();
        assert!(!result.passed);
        assert!(!result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_verification_system() {
        let mut system = VerificationSystem::default();
        
        let input = vec![1.0, 2.0, 3.0];
        let output = AlgorithmResult {
            data: vec![2.0, 4.0, 6.0],
            metadata: HashMap::new(),
            execution_time_ms: 10,
        };
        
        let result = system.verify_all("test_algorithm", &input, &output).await.unwrap();
        assert!(result.passed);
        
        // Check history
        let history = system.get_history("test_algorithm");
        assert!(history.is_some());
        assert_eq!(history.unwrap().len(), 1);
        
        // Check stats
        let stats = system.get_verification_stats("test_algorithm");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().total_verifications, 1);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.execution_time_ms, 0);
        assert!(metrics.memory_usage_mb.is_none());
    }

    #[test]
    fn test_data_constraint() {
        let constraint = DataConstraint {
            name: "test".to_string(),
            constraint_type: ConstraintType::Range { min: 0.0, max: 1.0 },
            parameters: HashMap::new(),
        };
        
        assert_eq!(constraint.name, "test");
        assert!(matches!(constraint.constraint_type, ConstraintType::Range { .. }));
    }
}