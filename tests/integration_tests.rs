//! Tests for the Loneboth AI framework

use loneboth-ai::{LonebothAI, Config, CoordinationMode, AlgorithmType};
use loneboth-ai::algorithms::{Algorithm, StaticAlgorithm, DynamicAlgorithm};
use loneboth-ai::coordination::CoordinationSystem;
use loneboth-ai::gpu::GpuAccelerator;
use loneboth-ai::verification::ConsistencyVerifier;

#[test]
fn test_loneboth-ai_creation() {
    let ai = LonebothAI::new();
    assert_eq!(ai.config().algorithm_type, AlgorithmType::Static);
    assert_eq!(ai.config().coordination_mode, CoordinationMode::Individual);
    assert!(ai.config().gpu_enabled);
    assert!(ai.config().verification_enabled);
}

#[test]
fn test_loneboth-ai_with_config() {
    let config = Config {
        gpu_enabled: false,
        coordination_mode: CoordinationMode::Group,
        verification_enabled: false,
        algorithm_type: AlgorithmType::Dynamic,
    };
    
    let ai = LonebothAI::with_config(config);
    assert_eq!(ai.config().algorithm_type, AlgorithmType::Dynamic);
    assert_eq!(ai.config().coordination_mode, CoordinationMode::Group);
    assert!(!ai.config().gpu_enabled);
    assert!(!ai.config().verification_enabled);
}

#[test]
fn test_basic_processing() {
    let ai = LonebothAI::new();
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = ai.process(&input);
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), input.len());
    
    // Results should be processed values, not identical to input
    assert_ne!(result, input);
}

#[test]
fn test_static_algorithm() {
    let algo = StaticAlgorithm::new();
    assert_eq!(algo.algorithm_type(), AlgorithmType::Static);
    assert_eq!(algo.name(), "StaticAlgorithm");
    assert!(algo.is_ready());
    
    let input = vec![1.0, 2.0, 3.0];
    let result = algo.process(&input);
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), input.len());
}

#[test]
fn test_dynamic_algorithm() {
    let algo = DynamicAlgorithm::new();
    assert_eq!(algo.algorithm_type(), AlgorithmType::Dynamic);
    assert_eq!(algo.name(), "DynamicAlgorithm");
    assert!(algo.is_ready());
    
    let input = vec![1.0, 2.0, 3.0];
    let result = algo.process(&input);
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), input.len());
}

#[test]
fn test_coordination_system() {
    let coord = CoordinationSystem::new(CoordinationMode::Individual);
    assert_eq!(coord.mode(), CoordinationMode::Individual);
    assert_eq!(coord.consensus_threshold(), 0.8);
    
    let algo = StaticAlgorithm::new();
    let input = vec![1.0, 2.0, 3.0];
    let result = coord.process(&algo, &input);
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), input.len());
}

#[test]
fn test_gpu_accelerator() {
    let gpu = GpuAccelerator::new();
    let info = gpu.info();
    
    assert!(!info.device_name.is_empty());
    
    let input = vec![1.0, 2.0, 3.0];
    let result = gpu.accelerate(&input);
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), input.len());
}

#[test]
fn test_consistency_verifier() {
    let verifier = ConsistencyVerifier::new(true);
    assert!(verifier.is_enabled());
    
    let valid_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = verifier.verify(&valid_data);
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.passed);
    assert!(result.confidence > 0.0);
}

#[test]
fn test_consistency_verifier_with_invalid_data() {
    let verifier = ConsistencyVerifier::new(true);
    
    let invalid_data = vec![f32::NAN, f32::INFINITY, 1.0];
    let result = verifier.verify(&invalid_data);
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(!result.passed); // Should fail due to NaN/Infinity
}

#[test]
fn test_different_coordination_modes() {
    let modes = vec![
        CoordinationMode::Individual,
        CoordinationMode::Group,
        CoordinationMode::Hybrid,
    ];
    
    for mode in modes {
        let config = Config {
            gpu_enabled: false,
            coordination_mode: mode,
            verification_enabled: true,
            algorithm_type: AlgorithmType::Static,
        };
        
        let ai = LonebothAI::with_config(config);
        let input = vec![1.0, 2.0, 3.0];
        let result = ai.process(&input);
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.len(), input.len());
    }
}

#[test]
fn test_empty_input() {
    let ai = LonebothAI::new();
    let input = vec![];
    let result = ai.process(&input);
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_large_input() {
    let ai = LonebothAI::new();
    let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let result = ai.process(&input);
    
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.len(), input.len());
}