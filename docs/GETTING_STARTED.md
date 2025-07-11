# Getting Started with LoneBoth AI

## Installation

### Prerequisites
- Rust 1.70+ (2024 edition support)
- ONNX Runtime (for GPU acceleration)
- DirectML (Windows only)

### Build from Source
```bash
git clone https://github.com/nanaisisi/loneboth_ai.git
cd loneboth_ai
cargo build --release
```

### GPU Support Setup

#### Windows (DirectML)
DirectML is automatically detected on Windows systems with DirectX 12 support.

#### Linux (ONNX Runtime)
```bash
# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xvf onnxruntime-linux-x64-1.16.0.tgz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/onnxruntime-linux-x64-1.16.0/lib
```

#### ARM Ubuntu
```bash
# Install ARM-optimized ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-aarch64-1.16.0.tgz
tar -xvf onnxruntime-linux-aarch64-1.16.0.tgz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/onnxruntime-linux-aarch64-1.16.0/lib
```

## Basic Usage

### Individual Algorithm Execution
```rust
use loneboth_ai::algorithm::Algorithm;
use loneboth_ai::coordination::IndividualCoordinator;

let mut coordinator = IndividualCoordinator::new();
let algorithm = Algorithm::load_static("my_algorithm")?;
let result = coordinator.execute(&algorithm, &input_data)?;
```

### Group Algorithm Coordination
```rust
use loneboth_ai::coordination::GroupCoordinator;

let mut coordinator = GroupCoordinator::new();
coordinator.add_algorithm("agent_1", algorithm_1);
coordinator.add_algorithm("agent_2", algorithm_2);
let consensus_result = coordinator.execute_consensus(&input_data)?;
```

### GPU-Accelerated Inference
```rust
use loneboth_ai::gpu::ONNXExecutor;

let mut executor = ONNXExecutor::new()?;
executor.load_model("model.onnx")?;
let output = executor.run_inference(&input_tensor)?;
```

### Dynamic Algorithm Loading
```rust
use loneboth_ai::runtime::DynamicLoader;

let loader = DynamicLoader::new();
let algorithm = loader.load_from_file("algorithm.dylib")?;
let result = algorithm.execute(&input_data)?;
```

## Configuration

### Algorithm Configuration
```toml
[algorithm]
name = "my_algorithm"
type = "static"
gpu_enabled = true
verification_enabled = true

[coordination]
type = "group"
consensus_threshold = 0.8
timeout_ms = 5000
```

### GPU Configuration
```toml
[gpu]
provider = "directml"  # or "cuda", "rocm", "cpu"
device_id = 0
memory_limit_mb = 2048
```

## Examples

### Example 1: Simple Algorithm Execution
```rust
use loneboth_ai::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let framework = LoneBothAI::new()?;
    
    // Load a simple algorithm
    let algorithm = framework.load_algorithm("examples/simple_add")?;
    
    // Execute with input data
    let input = vec![1.0, 2.0, 3.0];
    let result = framework.execute(&algorithm, &input)?;
    
    println!("Result: {:?}", result);
    Ok(())
}
```

### Example 2: Multi-Agent Coordination
```rust
use loneboth_ai::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut framework = LoneBothAI::new()?;
    
    // Create group coordinator
    let mut coordinator = GroupCoordinator::new();
    
    // Add multiple algorithms
    coordinator.add_algorithm("classifier", 
        framework.load_algorithm("models/classifier")?);
    coordinator.add_algorithm("regressor", 
        framework.load_algorithm("models/regressor")?);
    
    // Execute with consensus
    let input = load_test_data("data/input.json")?;
    let result = coordinator.execute_consensus(&input)?;
    
    println!("Consensus result: {:?}", result);
    Ok(())
}
```

### Example 3: GPU-Accelerated ONNX Model
```rust
use loneboth_ai::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let framework = LoneBothAI::new()?;
    
    // Initialize GPU executor
    let mut executor = framework.create_onnx_executor()?;
    executor.load_model("models/neural_network.onnx")?;
    
    // Prepare input tensor
    let input_tensor = Tensor::from_shape_and_data(
        vec![1, 3, 224, 224],
        load_image_data("input.jpg")?
    )?;
    
    // Run inference
    let output = executor.run_inference(&input_tensor)?;
    
    println!("Inference result: {:?}", output);
    Ok(())
}
```

## Testing

### Unit Tests
```bash
cargo test
```

### Integration Tests
```bash
cargo test --test integration
```

### Performance Tests
```bash
cargo test --test performance --release
```

## Debugging

### Enable Debug Logging
```bash
RUST_LOG=debug cargo run
```

### Performance Profiling
```bash
cargo flamegraph --bin loneboth_ai
```

### Memory Profiling
```bash
valgrind --tool=memcheck cargo run
```

## Common Issues

### Q: GPU not detected
**A:** Ensure proper GPU drivers are installed and ONNX Runtime/DirectML is correctly configured.

### Q: Algorithm loading fails
**A:** Check algorithm compatibility and ensure all dependencies are satisfied.

### Q: Performance issues
**A:** Enable release builds and consider GPU acceleration for compute-intensive operations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/nanaisisi/loneboth_ai/issues)
- Discussions: [GitHub Discussions](https://github.com/nanaisisi/loneboth_ai/discussions)