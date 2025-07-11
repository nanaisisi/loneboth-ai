//! GPU acceleration layer with ONNX Runtime and DirectML support

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("GPU initialization failed: {reason}")]
    InitializationFailed { reason: String },
    #[error("Model loading failed: {path} - {reason}")]
    ModelLoadFailed { path: String, reason: String },
    #[error("Inference failed: {reason}")]
    InferenceFailed { reason: String },
    #[error("Provider not available: {provider}")]
    ProviderNotAvailable { provider: String },
    #[error("Invalid tensor shape: expected {expected:?}, got {actual:?}")]
    InvalidTensorShape { expected: Vec<usize>, actual: Vec<usize> },
}

/// Supported execution providers for GPU acceleration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionProvider {
    /// CPU execution (fallback)
    CPU,
    /// NVIDIA CUDA
    CUDA,
    /// AMD ROCm
    ROCm,
    /// DirectML (Windows)
    DirectML,
    /// Apple Metal
    Metal,
    /// Vulkan compute
    Vulkan,
    /// Intel OpenVINO
    OpenVINO,
}

impl ExecutionProvider {
    pub fn is_gpu_accelerated(&self) -> bool {
        !matches!(self, ExecutionProvider::CPU)
    }
    
    pub fn get_name(&self) -> &'static str {
        match self {
            ExecutionProvider::CPU => "cpu",
            ExecutionProvider::CUDA => "cuda",
            ExecutionProvider::ROCm => "rocm",
            ExecutionProvider::DirectML => "directml",
            ExecutionProvider::Metal => "metal",
            ExecutionProvider::Vulkan => "vulkan",
            ExecutionProvider::OpenVINO => "openvino",
        }
    }
}

/// Tensor data structure for GPU operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub data_type: TensorDataType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorDataType {
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, GpuError> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(GpuError::InvalidTensorShape {
                expected: vec![expected_size],
                actual: vec![data.len()],
            });
        }
        
        Ok(Self {
            data,
            shape,
            data_type: TensorDataType::Float32,
        })
    }
    
    pub fn from_shape_and_data(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, GpuError> {
        Self::new(data, shape)
    }
    
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
            data_type: TensorDataType::Float32,
        }
    }
    
    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![1.0; size],
            shape,
            data_type: TensorDataType::Float32,
        }
    }
    
    pub fn get_size(&self) -> usize {
        self.shape.iter().product()
    }
    
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), GpuError> {
        let current_size = self.get_size();
        let new_size: usize = new_shape.iter().product();
        
        if current_size != new_size {
            return Err(GpuError::InvalidTensorShape {
                expected: vec![current_size],
                actual: vec![new_size],
            });
        }
        
        self.shape = new_shape;
        Ok(())
    }
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub device_id: i32,
    pub name: String,
    pub memory_mb: u64,
    pub compute_capability: Option<String>,
    pub provider: ExecutionProvider,
}

/// Core ONNX executor trait
#[async_trait]
pub trait ONNXExecutor: Send + Sync {
    async fn load_model(&mut self, model_path: &str) -> Result<(), GpuError>;
    async fn run_inference(&self, input: &Tensor) -> Result<Tensor, GpuError>;
    fn get_providers(&self) -> Vec<ExecutionProvider>;
    fn get_current_provider(&self) -> ExecutionProvider;
    async fn get_model_info(&self) -> Result<ModelInfo, GpuError>;
}

/// Model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub model_version: Option<String>,
}

/// ONNX Runtime implementation
pub struct OnnxRuntimeExecutor {
    provider: ExecutionProvider,
    device_id: i32,
    model_loaded: bool,
    model_path: Option<String>,
    memory_limit_mb: Option<u64>,
}

impl OnnxRuntimeExecutor {
    pub fn new(provider: ExecutionProvider, device_id: i32) -> Self {
        Self {
            provider,
            device_id,
            model_loaded: false,
            model_path: None,
            memory_limit_mb: None,
        }
    }
    
    pub fn with_memory_limit(mut self, limit_mb: u64) -> Self {
        self.memory_limit_mb = Some(limit_mb);
        self
    }
    
    /// Check if the specified provider is available on the system
    pub fn is_provider_available(&self, provider: &ExecutionProvider) -> bool {
        match provider {
            ExecutionProvider::CPU => true, // CPU is always available
            ExecutionProvider::CUDA => self.check_cuda_availability(),
            ExecutionProvider::DirectML => self.check_directml_availability(),
            ExecutionProvider::ROCm => self.check_rocm_availability(),
            ExecutionProvider::Metal => self.check_metal_availability(),
            ExecutionProvider::Vulkan => self.check_vulkan_availability(),
            ExecutionProvider::OpenVINO => self.check_openvino_availability(),
        }
    }
    
    fn check_cuda_availability(&self) -> bool {
        // In a real implementation, this would check for CUDA runtime
        cfg!(target_os = "linux") || cfg!(target_os = "windows")
    }
    
    fn check_directml_availability(&self) -> bool {
        // DirectML is Windows-only
        cfg!(target_os = "windows")
    }
    
    fn check_rocm_availability(&self) -> bool {
        // ROCm is primarily Linux
        cfg!(target_os = "linux")
    }
    
    fn check_metal_availability(&self) -> bool {
        // Metal is macOS-only
        cfg!(target_os = "macos")
    }
    
    fn check_vulkan_availability(&self) -> bool {
        // Vulkan is cross-platform but requires runtime check
        true // Simplified for prototype
    }
    
    fn check_openvino_availability(&self) -> bool {
        // OpenVINO availability check
        true // Simplified for prototype
    }
    
    /// Simulate model inference (placeholder implementation)
    async fn simulate_inference(&self, input: &Tensor) -> Result<Tensor, GpuError> {
        // Simulate processing time based on tensor size and provider
        let processing_time = match self.provider {
            ExecutionProvider::CPU => input.get_size() as u64 / 1000,
            _ => input.get_size() as u64 / 5000, // GPU is faster
        };
        
        if processing_time > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(processing_time)).await;
        }
        
        // Simulate some computation (element-wise square)
        let output_data: Vec<f32> = input.data.iter().map(|x| x * x).collect();
        
        Tensor::new(output_data, input.shape.clone())
    }
}

#[async_trait]
impl ONNXExecutor for OnnxRuntimeExecutor {
    async fn load_model(&mut self, model_path: &str) -> Result<(), GpuError> {
        // Simulate model loading
        log::info!("Loading ONNX model from: {}", model_path);
        
        // Check if provider is available
        if !self.is_provider_available(&self.provider) {
            return Err(GpuError::ProviderNotAvailable {
                provider: self.provider.get_name().to_string(),
            });
        }
        
        // Simulate loading time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        self.model_path = Some(model_path.to_string());
        self.model_loaded = true;
        
        log::info!("Model loaded successfully with provider: {}", self.provider.get_name());
        Ok(())
    }
    
    async fn run_inference(&self, input: &Tensor) -> Result<Tensor, GpuError> {
        if !self.model_loaded {
            return Err(GpuError::InferenceFailed {
                reason: "No model loaded".to_string(),
            });
        }
        
        log::debug!("Running inference with {} provider", self.provider.get_name());
        
        // Simulate inference
        self.simulate_inference(input).await
    }
    
    fn get_providers(&self) -> Vec<ExecutionProvider> {
        let mut providers = vec![ExecutionProvider::CPU];
        
        if self.check_cuda_availability() {
            providers.push(ExecutionProvider::CUDA);
        }
        if self.check_directml_availability() {
            providers.push(ExecutionProvider::DirectML);
        }
        if self.check_rocm_availability() {
            providers.push(ExecutionProvider::ROCm);
        }
        if self.check_metal_availability() {
            providers.push(ExecutionProvider::Metal);
        }
        if self.check_vulkan_availability() {
            providers.push(ExecutionProvider::Vulkan);
        }
        if self.check_openvino_availability() {
            providers.push(ExecutionProvider::OpenVINO);
        }
        
        providers
    }
    
    fn get_current_provider(&self) -> ExecutionProvider {
        self.provider.clone()
    }
    
    async fn get_model_info(&self) -> Result<ModelInfo, GpuError> {
        if !self.model_loaded {
            return Err(GpuError::InferenceFailed {
                reason: "No model loaded".to_string(),
            });
        }
        
        // Simulate model info (placeholder)
        Ok(ModelInfo {
            input_shapes: vec![vec![1, 3, 224, 224]], // Example image input
            output_shapes: vec![vec![1, 1000]], // Example classification output
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            model_version: Some("1.0.0".to_string()),
        })
    }
}

/// GPU device manager for discovering and managing GPU devices
pub struct GpuDeviceManager {
    devices: Vec<GpuDevice>,
    default_provider: ExecutionProvider,
}

impl GpuDeviceManager {
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            default_provider: ExecutionProvider::CPU,
        }
    }
    
    /// Discover available GPU devices
    pub async fn discover_devices(&mut self) -> Result<(), GpuError> {
        log::info!("Discovering GPU devices...");
        
        self.devices.clear();
        
        // Always add CPU device
        self.devices.push(GpuDevice {
            device_id: -1,
            name: "CPU".to_string(),
            memory_mb: 0, // System RAM is managed differently
            compute_capability: None,
            provider: ExecutionProvider::CPU,
        });
        
        // Simulate GPU device discovery
        #[cfg(target_os = "windows")]
        {
            // Simulate DirectML device
            self.devices.push(GpuDevice {
                device_id: 0,
                name: "DirectML GPU".to_string(),
                memory_mb: 4096,
                compute_capability: Some("DirectX 12".to_string()),
                provider: ExecutionProvider::DirectML,
            });
        }
        
        #[cfg(target_os = "linux")]
        {
            // Simulate CUDA device
            self.devices.push(GpuDevice {
                device_id: 0,
                name: "NVIDIA GPU".to_string(),
                memory_mb: 8192,
                compute_capability: Some("8.6".to_string()),
                provider: ExecutionProvider::CUDA,
            });
        }
        
        #[cfg(target_os = "macos")]
        {
            // Simulate Metal device
            self.devices.push(GpuDevice {
                device_id: 0,
                name: "Apple GPU".to_string(),
                memory_mb: 16384,
                compute_capability: Some("Metal 3".to_string()),
                provider: ExecutionProvider::Metal,
            });
        }
        
        // Set default provider to the first GPU device if available
        if let Some(gpu_device) = self.devices.iter().find(|d| d.provider.is_gpu_accelerated()) {
            self.default_provider = gpu_device.provider.clone();
        }
        
        log::info!("Discovered {} devices", self.devices.len());
        Ok(())
    }
    
    pub fn get_devices(&self) -> &[GpuDevice] {
        &self.devices
    }
    
    pub fn get_device(&self, device_id: i32) -> Option<&GpuDevice> {
        self.devices.iter().find(|d| d.device_id == device_id)
    }
    
    pub fn get_default_provider(&self) -> ExecutionProvider {
        self.default_provider.clone()
    }
    
    pub fn create_executor(&self, provider: Option<ExecutionProvider>, device_id: Option<i32>) -> Result<OnnxRuntimeExecutor, GpuError> {
        let provider = provider.unwrap_or_else(|| self.get_default_provider());
        let device_id = device_id.unwrap_or(0);
        
        // Verify device exists
        if provider.is_gpu_accelerated() {
            if self.get_device(device_id).is_none() {
                return Err(GpuError::InitializationFailed {
                    reason: format!("Device {} not found", device_id),
                });
            }
        }
        
        Ok(OnnxRuntimeExecutor::new(provider, device_id))
    }
}

impl Default for GpuDeviceManager {
    fn default() -> Self {
        let mut manager = Self::new();
        // Initialize with discovery in a blocking context
        tokio::task::block_in_place(|| {
            futures::executor::block_on(manager.discover_devices())
        }).unwrap_or_else(|e| {
            log::warn!("Failed to discover GPU devices: {}", e);
        });
        manager
    }
}

/// High-level GPU acceleration interface
pub struct GpuAccelerator {
    device_manager: GpuDeviceManager,
    executors: HashMap<String, Box<dyn ONNXExecutor>>,
}

impl GpuAccelerator {
    pub async fn new() -> Result<Self, GpuError> {
        let mut device_manager = GpuDeviceManager::new();
        device_manager.discover_devices().await?;
        
        Ok(Self {
            device_manager,
            executors: HashMap::new(),
        })
    }
    
    pub fn get_available_providers(&self) -> Vec<ExecutionProvider> {
        self.device_manager.get_devices()
            .iter()
            .map(|d| d.provider.clone())
            .collect()
    }
    
    pub fn get_devices(&self) -> &[GpuDevice] {
        self.device_manager.get_devices()
    }
    
    pub async fn create_executor(&mut self, name: String, provider: Option<ExecutionProvider>) -> Result<(), GpuError> {
        let executor = self.device_manager.create_executor(provider, None)?;
        self.executors.insert(name, Box::new(executor));
        Ok(())
    }
    
    pub async fn load_model(&mut self, executor_name: &str, model_path: &str) -> Result<(), GpuError> {
        match self.executors.get_mut(executor_name) {
            Some(executor) => executor.load_model(model_path).await,
            None => Err(GpuError::InitializationFailed {
                reason: format!("Executor {} not found", executor_name),
            }),
        }
    }
    
    pub async fn run_inference(&self, executor_name: &str, input: &Tensor) -> Result<Tensor, GpuError> {
        match self.executors.get(executor_name) {
            Some(executor) => executor.run_inference(input).await,
            None => Err(GpuError::InitializationFailed {
                reason: format!("Executor {} not found", executor_name),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data.clone(), shape.clone()).unwrap();
        
        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.get_size(), 4);
    }

    #[test]
    fn test_tensor_reshape() {
        let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        tensor.reshape(vec![1, 4]).unwrap();
        
        assert_eq!(tensor.shape, vec![1, 4]);
        assert_eq!(tensor.data.len(), 4);
    }

    #[test]
    fn test_tensor_invalid_shape() {
        let result = Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_execution_provider_methods() {
        let provider = ExecutionProvider::CUDA;
        assert!(provider.is_gpu_accelerated());
        assert_eq!(provider.get_name(), "cuda");
        
        let cpu_provider = ExecutionProvider::CPU;
        assert!(!cpu_provider.is_gpu_accelerated());
        assert_eq!(cpu_provider.get_name(), "cpu");
    }

    #[tokio::test]
    async fn test_onnx_executor() {
        let mut executor = OnnxRuntimeExecutor::new(ExecutionProvider::CPU, 0);
        
        // Test model loading
        let result = executor.load_model("test_model.onnx").await;
        assert!(result.is_ok());
        
        // Test inference
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = executor.run_inference(&input).await.unwrap();
        
        assert_eq!(output.shape, input.shape);
        assert_eq!(output.data, vec![1.0, 4.0, 9.0, 16.0]); // Squared values
    }

    #[tokio::test]
    async fn test_gpu_device_manager() {
        let mut manager = GpuDeviceManager::new();
        manager.discover_devices().await.unwrap();
        
        let devices = manager.get_devices();
        assert!(!devices.is_empty());
        
        // CPU device should always be available
        assert!(devices.iter().any(|d| matches!(d.provider, ExecutionProvider::CPU)));
    }

    #[tokio::test]
    async fn test_gpu_accelerator() {
        let mut accelerator = GpuAccelerator::new().await.unwrap();
        
        // Create executor
        accelerator.create_executor("test".to_string(), Some(ExecutionProvider::CPU)).await.unwrap();
        
        // Load model
        accelerator.load_model("test", "test_model.onnx").await.unwrap();
        
        // Run inference
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = accelerator.run_inference("test", &input).await.unwrap();
        
        assert_eq!(output.shape, input.shape);
    }
}