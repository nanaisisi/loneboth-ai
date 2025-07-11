# LoneBoth AI Framework API Reference

## Core Framework

### `LoneBothAI`

The main framework entry point.

```rust
use loneboth_ai::prelude::*;

let framework = LoneBothAI::new()?;
```

#### Methods

- `new() -> Result<Self, Box<dyn Error>>` - Initialize the framework
- `is_initialized(&self) -> bool` - Check if framework is ready

## Algorithm Management

### `Algorithm` Trait

Core trait for all algorithms in the framework.

```rust
#[async_trait]
pub trait Algorithm: Send + Sync {
    async fn execute(&self, input: &[f32]) -> Result<AlgorithmResult, AlgorithmError>;
    fn get_type(&self) -> AlgorithmType;
    fn get_name(&self) -> &str;
    fn get_version(&self) -> &str;
}
```

### `StaticAlgorithm`

Compile-time algorithms for maximum performance.

```rust
let algorithm = StaticAlgorithm::new(
    "my_algorithm".to_string(),
    "1.0.0".to_string(),
    |input| input.iter().map(|x| x * 2.0).collect(),
);
```

### `DynamicAlgorithm`

Runtime-loadable algorithms via dylib.

```rust
let algorithm = DynamicAlgorithm::new(
    "dynamic_algo".to_string(),
    "1.0.0".to_string(),
);
```

### `FuzzyAlgorithm`

Adaptive algorithms that learn from execution patterns.

```rust
let algorithm = FuzzyAlgorithm::new(
    "adaptive_algo".to_string(),
    "1.0.0".to_string(),
    0.5, // adaptation factor
);
```

### `AlgorithmRegistry`

Central registry for managing algorithm instances.

```rust
let mut registry = AlgorithmRegistry::default();
registry.register(Box::new(algorithm));
let algorithm = registry.get("algorithm_name");
```

## Coordination System

### `IndividualCoordinator`

Sequential algorithm execution with minimal overhead.

```rust
let mut coordinator = IndividualCoordinator::new();
coordinator.add_algorithm("algo".to_string(), Box::new(algorithm));
coordinator.set_current_algorithm("algo".to_string())?;
let result = coordinator.execute(&input_data).await?;
```

### `GroupCoordinator`

Multi-agent coordination with consensus mechanisms.

```rust
let mut coordinator = GroupCoordinator::new()
    .with_consensus_threshold(0.8)
    .with_timeout(5000);

coordinator.add_algorithm("agent1".to_string(), Box::new(algorithm1));
coordinator.add_algorithm("agent2".to_string(), Box::new(algorithm2));

let result = coordinator.execute_consensus(&input_data).await?;
```

#### Methods

- `execute_parallel(&self, input: &[f32])` - Execute all algorithms in parallel
- `execute_consensus(&self, input: &[f32])` - Execute with consensus verification
- `with_consensus_threshold(threshold: f32)` - Set consensus threshold (0.0-1.0)
- `with_timeout(timeout_ms: u64)` - Set execution timeout

## GPU Acceleration

### `GpuAccelerator`

High-level interface for GPU-accelerated operations.

```rust
let mut accelerator = GpuAccelerator::new().await?;
accelerator.create_executor("my_executor".to_string(), Some(ExecutionProvider::CUDA)).await?;
accelerator.load_model("my_executor", "model.onnx").await?;

let output = accelerator.run_inference("my_executor", &input_tensor).await?;
```

### `Tensor`

Multi-dimensional data structure for GPU operations.

```rust
let tensor = Tensor::new(
    vec![1.0, 2.0, 3.0, 4.0],
    vec![2, 2] // shape: 2x2 matrix
)?;

let zeros = Tensor::zeros(vec![3, 3]);
let ones = Tensor::ones(vec![2, 4]);
```

#### Methods

- `new(data: Vec<f32>, shape: Vec<usize>)` - Create tensor with data and shape
- `zeros(shape: Vec<usize>)` - Create zero-filled tensor
- `ones(shape: Vec<usize>)` - Create one-filled tensor
- `reshape(&mut self, new_shape: Vec<usize>)` - Reshape tensor
- `get_size(&self) -> usize` - Get total number of elements

### `ExecutionProvider`

Enum representing different GPU execution providers.

```rust
pub enum ExecutionProvider {
    CPU,        // CPU execution (fallback)
    CUDA,       // NVIDIA CUDA
    ROCm,       // AMD ROCm
    DirectML,   // DirectML (Windows)
    Metal,      // Apple Metal
    Vulkan,     // Vulkan compute
    OpenVINO,   // Intel OpenVINO
}
```

### `ONNXExecutor` Trait

Interface for ONNX model execution.

```rust
#[async_trait]
pub trait ONNXExecutor: Send + Sync {
    async fn load_model(&mut self, model_path: &str) -> Result<(), GpuError>;
    async fn run_inference(&self, input: &Tensor) -> Result<Tensor, GpuError>;
    fn get_providers(&self) -> Vec<ExecutionProvider>;
    fn get_current_provider(&self) -> ExecutionProvider;
    async fn get_model_info(&self) -> Result<ModelInfo, GpuError>;
}
```

## Runtime System

### `RuntimeManager`

Manages dynamic algorithm loading and execution.

```rust
let mut manager = RuntimeManager::new();
manager.initialize().await?;

let active_plugins = manager.list_active_plugins();
let result = manager.execute_plugin("plugin_name", &input_data).await?;
```

### `DynamicLoader`

Loads algorithms from shared libraries.

```rust
let mut loader = DynamicLoader::new();
loader.add_plugin_directory("./plugins");
let discovered = loader.discover_plugins().await?;
let plugin = loader.load_from_file("algorithm.so").await?;
```

### `AlgorithmMetadata`

Metadata structure for dynamic algorithms.

```rust
pub struct AlgorithmMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub api_version: String,
    pub dependencies: Vec<String>,
    pub capabilities: Vec<String>,
}
```

## Verification System

### `VerificationSystem`

Comprehensive verification for algorithm integrity.

```rust
let mut system = VerificationSystem::default();
let result = system.verify_all("algorithm_name", &input, &output).await?;

println!("Verification passed: {}", result.passed);
println!("Integrity score: {}", result.integrity_score);
```

### `AlgorithmIntegrityVerifier`

Verifies algorithm integrity through checksums and performance monitoring.

```rust
let mut verifier = AlgorithmIntegrityVerifier::new();
verifier.add_checksum("algorithm_name".to_string(), "expected_hash".to_string());
verifier.add_performance_threshold("algorithm_name".to_string(), 1000); // 1 second
```

### `DataIntegrityVerifier`

Validates input/output data integrity.

```rust
let mut verifier = DataIntegrityVerifier::new();
verifier.add_input_constraint(DataConstraint {
    name: "range_check".to_string(),
    constraint_type: ConstraintType::Range { min: 0.0, max: 1.0 },
    parameters: HashMap::new(),
});
```

### `VerificationResult`

Result structure containing verification details.

```rust
pub struct VerificationResult {
    pub passed: bool,
    pub checks_performed: Vec<String>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub performance_metrics: PerformanceMetrics,
    pub integrity_score: f32,
}
```

## Error Types

### `AlgorithmError`

Errors related to algorithm execution.

```rust
pub enum AlgorithmError {
    NotFound { name: String },
    ExecutionFailed { reason: String },
    InvalidType { algorithm_type: String },
}
```

### `CoordinationError`

Errors in coordination systems.

```rust
pub enum CoordinationError {
    AlgorithmNotFound { name: String },
    CoordinationFailed { reason: String },
    ConsensusNotReached { threshold: f32, actual: f32 },
    Timeout { timeout_ms: u64 },
}
```

### `GpuError`

GPU-related errors.

```rust
pub enum GpuError {
    InitializationFailed { reason: String },
    ModelLoadFailed { path: String, reason: String },
    InferenceFailed { reason: String },
    ProviderNotAvailable { provider: String },
    InvalidTensorShape { expected: Vec<usize>, actual: Vec<usize> },
}
```

### `RuntimeError`

Runtime system errors.

```rust
pub enum RuntimeError {
    LibraryLoadFailed { path: String, reason: String },
    SymbolNotFound { symbol: String, library: String },
    VersionMismatch { expected: String, actual: String },
    ValidationFailed { reason: String },
    PluginNotFound { name: String },
    RuntimeError { reason: String },
}
```

### `VerificationError`

Verification system errors.

```rust
pub enum VerificationError {
    ChecksumMismatch { expected: String, actual: String },
    ValidationFailed { reason: String },
    ConsistencyFailed { reason: String },
    PerformanceRegression { current: u64, threshold: u64 },
    DataIntegrityViolation { field: String },
}
```

## Configuration

### Features

Enable specific functionality through Cargo features:

```toml
[features]
default = ["gpu"]
gpu = []           # GPU acceleration support
onnx = []          # ONNX Runtime integration
directml = []      # DirectML support (Windows)
```

### Environment Variables

- `RUST_LOG` - Set logging level (debug, info, warn, error)
- `LONEBOTH_PLUGIN_PATH` - Additional plugin search directories

## Examples

See the `examples/` directory for comprehensive usage examples:

- `comprehensive_demo.rs` - Complete framework demonstration
- Run with: `cargo run --example comprehensive_demo`

## Thread Safety

All major components are designed to be thread-safe:

- `Algorithm` trait requires `Send + Sync`
- `Coordinator` implementations use async/await
- `RuntimeManager` uses `Arc<RwLock<>>` for shared state
- GPU operations are internally synchronized

## Performance Considerations

- Static algorithms have zero-cost abstractions
- Dynamic algorithms have plugin loading overhead
- GPU operations use async execution to avoid blocking
- Verification can be disabled in production for performance
- Memory pooling is used for frequent allocations

## Platform Support

- **Linux**: Full support with CUDA/ROCm
- **Windows**: DirectML support + CPU fallback
- **macOS**: Metal support + CPU fallback
- **ARM Ubuntu**: Optimized ONNX Runtime builds