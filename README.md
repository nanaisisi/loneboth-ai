# LoneBoth AI Framework

A comprehensive AI framework for algorithm coordination and GPU-accelerated machine learning operations.

## Overview

LoneBoth AI is designed to provide a flexible and efficient platform for:
- Individual and group algorithm coordination
- GPU acceleration through DirectML and ONNX Runtime
- Static and dynamic algorithm management
- Cross-platform support (ARM Ubuntu and x86)

## Core Features

### Algorithm Coordination
- **Individual/Group Coordination (単体/群体協調)**: Support for both single-agent and multi-agent algorithm coordination
- **Algorithm Definition (アルゴリズム定義)**: Flexible framework for defining custom algorithms
- **Algorithm Extensions (アルゴリズム拡張)**: 
  - Static (固定): Fixed algorithm implementations
  - Dynamic (ファジー): Fuzzy and adaptive algorithm implementations

### GPU Acceleration
- **ONNX Runtime Integration**: High-performance inference engine
- **DirectML Support**: Hardware-accelerated ML operations on Windows
- **Cross-platform GPU Support**: Optimized for various hardware configurations

### Algorithm Management
- **Static Algorithms (静的)**: Internal, compiled algorithms for maximum performance
- **Dynamic Algorithms (動的)**: Runtime-loadable algorithms via dylib/cdylib
- **Variable Algorithms (可変アルゴリズム)**: Adaptive algorithms that can modify their behavior
- **Consistency Verification (整合性確認)**: Built-in verification system for algorithm integrity

## Architecture

The framework is built with modularity and performance in mind:

```
loneboth_ai/
├── core/           # Core framework components
├── algorithms/     # Algorithm implementations
├── coordination/   # Individual/Group coordination logic
├── gpu/           # GPU acceleration modules
├── runtime/       # Dynamic algorithm runtime
└── verification/  # Consistency verification system
```

## Platform Support

- **Windows**: DirectML acceleration
- **Linux**: ONNX Runtime with CUDA/ROCm
- **ARM Ubuntu**: Optimized for ARM64 architecture
- **Cross-platform**: Rust-based for maximum portability

## Getting Started

```bash
cargo build --release
cargo test
cargo run
```

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
