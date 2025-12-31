# API Documentation

## Overview

The Loneboth AI framework provides a comprehensive set of APIs for AI processing with support for multiple coordination modes, algorithm types, and GPU acceleration.

## Main API

### `LonebothAI`

The main entry point for the framework.

#### Constructor Methods

```rust
// Create with default configuration
let ai = LonebothAI::new();

// Create with custom configuration
let config = Config {
    gpu_enabled: true,
    coordination_mode: CoordinationMode::Group,
    verification_enabled: true,
    algorithm_type: AlgorithmType::Dynamic,
};
let ai = LonebothAI::with_config(config);
```

#### Processing Methods

```rust
// Process input data
let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let result = ai.process(&input)?;
```

#### Configuration Methods

```rust
// Get current configuration
let config = ai.config();
```

## Configuration

### `Config`

Main configuration structure for the framework.

```rust
pub struct Config {
    pub gpu_enabled: bool,
    pub coordination_mode: CoordinationMode,
    pub verification_enabled: bool,
    pub algorithm_type: AlgorithmType,
}
```

### `CoordinationMode`

Defines how algorithms are coordinated.

```rust
pub enum CoordinationMode {
    Individual,  // Single algorithm execution
    Group,       // Multi-algorithm coordination
    Hybrid,      // Combined individual + group
}
```

### `AlgorithmType`

Defines the type of algorithm to use.

```rust
pub enum AlgorithmType {
    Static,    // Fixed, pre-defined algorithms
    Dynamic,   // Adaptive algorithms
    Variable,  // Runtime configurable
}
```

## Algorithm API

### `Algorithm` Trait

Common interface for all algorithms.

```rust
pub trait Algorithm {
    fn process(&self, input: &[f32]) -> Result<Vec<f32>>;
    fn algorithm_type(&self) -> AlgorithmType;
    fn name(&self) -> &str;
    fn is_ready(&self) -> bool;
}
```

### Built-in Algorithms

#### `StaticAlgorithm`

```rust
let algo = StaticAlgorithm::new();
let result = algo.process(&input_data)?;
```

#### `DynamicAlgorithm`

```rust
let mut algo = DynamicAlgorithm::new();
algo.set_adaptation_factor(1.5);
let result = algo.process(&input_data)?;
```

## Coordination API

### `CoordinationSystem`

Manages coordination between algorithms.

```rust
let coord = CoordinationSystem::new(CoordinationMode::Group);
let result = coord.process(&algorithm, &input)?;

// Set consensus threshold
coord.set_consensus_threshold(0.9);
```

## GPU Acceleration API

### `GpuAccelerator`

Provides GPU acceleration capabilities.

```rust
let gpu = GpuAccelerator::new();
let accelerated_result = gpu.accelerate(&input)?;

// Check GPU info
let info = gpu.info();
println!("GPU: {}", info.device_name);
```

### `GpuBackend`

Available GPU backends.

```rust
pub enum GpuBackend {
    OnnxRuntime,  // Cross-platform ONNX Runtime
    DirectML,     // Windows DirectML
    Cpu,          // CPU fallback
}
```

## Verification API

### `ConsistencyVerifier`

Provides result verification and validation.

```rust
let verifier = ConsistencyVerifier::new(true);
let verification_result = verifier.verify(&result)?;

if verification_result.passed {
    println!("Verification passed with confidence: {}", verification_result.confidence);
}
```

### `VerificationResult`

Result of verification process.

```rust
pub struct VerificationResult {
    pub passed: bool,
    pub confidence: f32,
    pub message: String,
}
```

## Error Handling

The framework uses the standard Rust `Result` type for error handling:

```rust
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
```

## Usage Examples

### Basic Usage

```rust
use loneboth-ai::{LonebothAI, Config, CoordinationMode, AlgorithmType};

// Create AI instance
let ai = LonebothAI::new();

// Process data
let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let result = ai.process(&input)?;

println!("Result: {:?}", result);
```

### Advanced Configuration

```rust
// Custom configuration
let config = Config {
    gpu_enabled: true,
    coordination_mode: CoordinationMode::Hybrid,
    verification_enabled: true,
    algorithm_type: AlgorithmType::Dynamic,
};

let ai = LonebothAI::with_config(config);
let result = ai.process(&input)?;
```

### Algorithm Comparison

```rust
// Compare different algorithms
let algorithms = vec![
    AlgorithmType::Static,
    AlgorithmType::Dynamic,
];

for algo_type in algorithms {
    let config = Config {
        algorithm_type: algo_type,
        ..Default::default()
    };

    let ai = LonebothAI::with_config(config);
    let result = ai.process(&input)?;

    println!("{:?} result: {:?}", algo_type, result);
}
```

## Performance Considerations

- **GPU Acceleration**: Enable GPU acceleration for large datasets
- **Coordination Mode**: Individual mode is fastest, Group mode provides better consensus
- **Verification**: Disable verification for performance-critical applications
- **Algorithm Type**: Static algorithms are fastest, Dynamic algorithms are most flexible

## Platform Support

- **Linux**: Full support including GPU acceleration
- **Windows**: Full support with DirectML acceleration
- **macOS**: CPU processing with limited GPU support
- **ARM**: Supported on ARM64 platforms
