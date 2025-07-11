//! Consistency verification system
//! 
//! Provides algorithm result validation and integrity checking.

use crate::Result;

/// Consistency verification implementation
pub struct ConsistencyVerifier {
    enabled: bool,
    tolerance: f32,
}

impl ConsistencyVerifier {
    /// Create a new consistency verifier
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            tolerance: 1e-6, // Default tolerance for floating-point comparisons
        }
    }

    /// Verify result consistency
    pub fn verify(&self, result: &[f32]) -> Result<VerificationResult> {
        if !self.enabled {
            return Ok(VerificationResult::new(true, 1.0, "Verification disabled"));
        }

        // Perform various consistency checks
        let checks = vec![
            self.check_nan_inf(result),
            self.check_range_validity(result),
            self.check_statistical_validity(result),
            self.check_pattern_consistency(result),
        ];

        // Calculate overall verification result
        let passed_checks = checks.iter().filter(|check| check.passed).count();
        let total_checks = checks.len();
        let confidence = passed_checks as f32 / total_checks as f32;

        let overall_passed = confidence >= 0.8; // 80% of checks must pass
        let message = if overall_passed {
            format!("Verification passed: {}/{} checks", passed_checks, total_checks)
        } else {
            format!("Verification failed: {}/{} checks", passed_checks, total_checks)
        };

        Ok(VerificationResult::new(overall_passed, confidence, &message))
    }

    /// Check for NaN and infinite values
    fn check_nan_inf(&self, result: &[f32]) -> VerificationCheck {
        let has_nan_inf = result.iter().any(|&x| x.is_nan() || x.is_infinite());
        
        VerificationCheck {
            name: "NaN/Inf Check".to_string(),
            passed: !has_nan_inf,
            message: if has_nan_inf {
                "Result contains NaN or infinite values".to_string()
            } else {
                "No NaN or infinite values found".to_string()
            },
        }
    }

    /// Check if values are within reasonable range
    fn check_range_validity(&self, result: &[f32]) -> VerificationCheck {
        let max_value = result.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_value = result.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        
        // Check if values are within a reasonable range
        let range_valid = max_value < 1e6 && min_value > -1e6;
        
        VerificationCheck {
            name: "Range Validity Check".to_string(),
            passed: range_valid,
            message: if range_valid {
                format!("Values in valid range: [{:.3}, {:.3}]", min_value, max_value)
            } else {
                format!("Values outside valid range: [{:.3}, {:.3}]", min_value, max_value)
            },
        }
    }

    /// Check statistical properties
    fn check_statistical_validity(&self, result: &[f32]) -> VerificationCheck {
        if result.is_empty() {
            return VerificationCheck {
                name: "Statistical Validity Check".to_string(),
                passed: false,
                message: "Empty result array".to_string(),
            };
        }

        let sum: f32 = result.iter().sum();
        let mean = sum / result.len() as f32;
        
        // Calculate variance
        let variance: f32 = result.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / result.len() as f32;
        
        // Check if variance is reasonable (not too high or too low)
        let variance_valid = variance > 1e-10 && variance < 1e10;
        
        VerificationCheck {
            name: "Statistical Validity Check".to_string(),
            passed: variance_valid,
            message: if variance_valid {
                format!("Statistical properties valid: mean={:.3}, variance={:.3}", mean, variance)
            } else {
                format!("Invalid statistical properties: mean={:.3}, variance={:.3}", mean, variance)
            },
        }
    }

    /// Check pattern consistency
    fn check_pattern_consistency(&self, result: &[f32]) -> VerificationCheck {
        if result.len() < 2 {
            return VerificationCheck {
                name: "Pattern Consistency Check".to_string(),
                passed: true,
                message: "Insufficient data for pattern check".to_string(),
            };
        }

        // Check for sudden jumps or discontinuities
        let mut max_diff = 0.0f32;
        for i in 1..result.len() {
            let diff = (result[i] - result[i-1]).abs();
            max_diff = max_diff.max(diff);
        }

        // Check if maximum difference is reasonable
        let pattern_consistent = max_diff < 1000.0; // Arbitrary threshold
        
        VerificationCheck {
            name: "Pattern Consistency Check".to_string(),
            passed: pattern_consistent,
            message: if pattern_consistent {
                format!("Pattern consistent: max_diff={:.3}", max_diff)
            } else {
                format!("Pattern inconsistent: max_diff={:.3}", max_diff)
            },
        }
    }

    /// Set tolerance for floating-point comparisons
    pub fn set_tolerance(&mut self, tolerance: f32) {
        self.tolerance = tolerance;
    }

    /// Get current tolerance
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Enable or disable verification
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if verification is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Verification result
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub passed: bool,
    pub confidence: f32,
    pub message: String,
}

impl VerificationResult {
    /// Create a new verification result
    pub fn new(passed: bool, confidence: f32, message: &str) -> Self {
        Self {
            passed,
            confidence,
            message: message.to_string(),
        }
    }
}

/// Individual verification check
#[derive(Debug, Clone)]
pub struct VerificationCheck {
    pub name: String,
    pub passed: bool,
    pub message: String,
}

impl VerificationCheck {
    /// Create a new verification check
    pub fn new(name: &str, passed: bool, message: &str) -> Self {
        Self {
            name: name.to_string(),
            passed,
            message: message.to_string(),
        }
    }
}