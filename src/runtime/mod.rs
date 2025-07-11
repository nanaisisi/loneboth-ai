//! Runtime system for dynamic algorithm loading and management

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

use crate::algorithm::{Algorithm, AlgorithmResult, AlgorithmError, AlgorithmType};

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("Failed to load library: {path} - {reason}")]
    LibraryLoadFailed { path: String, reason: String },
    #[error("Symbol not found: {symbol} in {library}")]
    SymbolNotFound { symbol: String, library: String },
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: String, actual: String },
    #[error("Plugin validation failed: {reason}")]
    ValidationFailed { reason: String },
    #[error("Plugin not found: {name}")]
    PluginNotFound { name: String },
    #[error("Runtime error: {reason}")]
    RuntimeError { reason: String },
}

/// Dynamic algorithm interface for plugins
#[async_trait]
pub trait DynamicAlgorithm: Send + Sync {
    async fn initialize(&mut self) -> Result<(), RuntimeError>;
    async fn execute(&self, input: &[u8]) -> Result<Vec<u8>, RuntimeError>;
    async fn cleanup(&mut self) -> Result<(), RuntimeError>;
    fn get_metadata(&self) -> AlgorithmMetadata;
}

/// Algorithm metadata for plugin information
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for AlgorithmMetadata {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            version: "0.0.0".to_string(),
            description: "No description".to_string(),
            author: "Unknown".to_string(),
            license: "Unknown".to_string(),
            api_version: "1.0.0".to_string(),
            dependencies: Vec::new(),
            capabilities: Vec::new(),
        }
    }
}

/// Plugin container for managing dynamic algorithms
pub struct Plugin {
    metadata: AlgorithmMetadata,
    algorithm: Box<dyn DynamicAlgorithm>,
    library_path: PathBuf,
    loaded_at: std::time::Instant,
    execution_count: u64,
}

impl Plugin {
    pub fn new(
        metadata: AlgorithmMetadata,
        algorithm: Box<dyn DynamicAlgorithm>,
        library_path: PathBuf,
    ) -> Self {
        Self {
            metadata,
            algorithm,
            library_path,
            loaded_at: std::time::Instant::now(),
            execution_count: 0,
        }
    }
    
    pub fn get_metadata(&self) -> &AlgorithmMetadata {
        &self.metadata
    }
    
    pub fn get_library_path(&self) -> &Path {
        &self.library_path
    }
    
    pub fn get_uptime(&self) -> std::time::Duration {
        self.loaded_at.elapsed()
    }
    
    pub fn get_execution_count(&self) -> u64 {
        self.execution_count
    }
    
    pub async fn execute(&mut self, input: &[u8]) -> Result<Vec<u8>, RuntimeError> {
        let result = self.algorithm.execute(input).await;
        if result.is_ok() {
            self.execution_count += 1;
        }
        result
    }
}

/// Wrapper to adapt DynamicAlgorithm to Algorithm trait
pub struct DynamicAlgorithmWrapper {
    plugin: Arc<RwLock<Plugin>>,
}

impl DynamicAlgorithmWrapper {
    pub fn new(plugin: Plugin) -> Self {
        Self {
            plugin: Arc::new(RwLock::new(plugin)),
        }
    }
    
    pub async fn get_metadata(&self) -> AlgorithmMetadata {
        let plugin = self.plugin.read().await;
        plugin.get_metadata().clone()
    }
}

#[async_trait]
impl Algorithm for DynamicAlgorithmWrapper {
    async fn execute(&self, input: &[f32]) -> Result<AlgorithmResult, AlgorithmError> {
        let start = std::time::Instant::now();
        
        // Convert f32 slice to bytes
        let input_bytes: Vec<u8> = input.iter()
            .flat_map(|f| f.to_le_bytes().to_vec())
            .collect();
        
        // Execute through plugin
        let mut plugin = self.plugin.write().await;
        let output_bytes = plugin.execute(&input_bytes).await
            .map_err(|e| AlgorithmError::ExecutionFailed { 
                reason: e.to_string() 
            })?;
        
        // Convert bytes back to f32
        let output_data: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(bytes)
            })
            .collect();
        
        let execution_time = start.elapsed().as_millis() as u64;
        
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "dynamic".to_string());
        metadata.insert("plugin_path".to_string(), 
                        plugin.get_library_path().to_string_lossy().to_string());
        metadata.insert("execution_count".to_string(), 
                        plugin.get_execution_count().to_string());
        
        Ok(AlgorithmResult {
            data: output_data,
            metadata,
            execution_time_ms: execution_time,
        })
    }
    
    fn get_type(&self) -> AlgorithmType {
        AlgorithmType::Dynamic
    }
    
    fn get_name(&self) -> &str {
        // This is a simplification - in a real implementation, we'd need to handle this differently
        "dynamic_algorithm"
    }
    
    fn get_version(&self) -> &str {
        "1.0.0"
    }
}

/// Dynamic algorithm loader for managing plugins
pub struct DynamicLoader {
    plugins: HashMap<String, Arc<RwLock<Plugin>>>,
    plugin_directories: Vec<PathBuf>,
    api_version: String,
}

impl DynamicLoader {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            plugin_directories: vec![
                PathBuf::from("./plugins"),
                PathBuf::from("/usr/local/lib/loneboth_ai/plugins"),
                PathBuf::from("~/.loneboth_ai/plugins"),
            ],
            api_version: "1.0.0".to_string(),
        }
    }
    
    pub fn add_plugin_directory<P: AsRef<Path>>(&mut self, path: P) {
        self.plugin_directories.push(path.as_ref().to_path_buf());
    }
    
    /// Load plugin from file (placeholder implementation)
    pub async fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<Arc<RwLock<Plugin>>, RuntimeError> {
        let path = path.as_ref();
        
        log::info!("Loading plugin from: {}", path.display());
        
        // In a real implementation, this would use libloading to load the dylib
        // For the prototype, we'll create a mock plugin
        let mock_plugin = self.create_mock_plugin(path)?;
        
        let plugin_name = mock_plugin.get_metadata().name.clone();
        let plugin = Arc::new(RwLock::new(mock_plugin));
        
        self.plugins.insert(plugin_name, plugin.clone());
        
        log::info!("Plugin loaded successfully: {}", path.display());
        Ok(plugin)
    }
    
    /// Discover and load plugins from configured directories
    pub async fn discover_plugins(&mut self) -> Result<Vec<String>, RuntimeError> {
        let mut loaded_plugins = Vec::new();
        
        for directory in &self.plugin_directories {
            if !directory.exists() {
                log::debug!("Plugin directory does not exist: {}", directory.display());
                continue;
            }
            
            log::info!("Scanning plugin directory: {}", directory.display());
            
            // In a real implementation, this would scan for .so/.dll/.dylib files
            // For the prototype, we'll create some mock plugins
            let mock_plugins = self.create_mock_plugins_for_directory(directory)?;
            
            for plugin in mock_plugins {
                let plugin_name = plugin.get_metadata().name.clone();
                self.plugins.insert(plugin_name.clone(), Arc::new(RwLock::new(plugin)));
                loaded_plugins.push(plugin_name);
            }
        }
        
        log::info!("Discovered {} plugins", loaded_plugins.len());
        Ok(loaded_plugins)
    }
    
    /// Get plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<Arc<RwLock<Plugin>>> {
        self.plugins.get(name).cloned()
    }
    
    /// List all loaded plugins
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.keys().cloned().collect()
    }
    
    /// Unload plugin
    pub async fn unload_plugin(&mut self, name: &str) -> Result<(), RuntimeError> {
        match self.plugins.remove(name) {
            Some(plugin) => {
                let mut plugin = plugin.write().await;
                plugin.algorithm.cleanup().await?;
                log::info!("Plugin unloaded: {}", name);
                Ok(())
            }
            None => Err(RuntimeError::PluginNotFound { name: name.to_string() })
        }
    }
    
    /// Validate plugin compatibility
    fn validate_plugin(&self, metadata: &AlgorithmMetadata) -> Result<(), RuntimeError> {
        // Check API version compatibility
        if metadata.api_version != self.api_version {
            return Err(RuntimeError::VersionMismatch {
                expected: self.api_version.clone(),
                actual: metadata.api_version.clone(),
            });
        }
        
        // Check for required fields
        if metadata.name.is_empty() {
            return Err(RuntimeError::ValidationFailed {
                reason: "Plugin name cannot be empty".to_string(),
            });
        }
        
        // Check for version format (simplified)
        if !metadata.version.contains('.') {
            return Err(RuntimeError::ValidationFailed {
                reason: "Invalid version format".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Create mock plugin for prototype (placeholder)
    fn create_mock_plugin(&self, path: &Path) -> Result<Plugin, RuntimeError> {
        let filename = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        
        let metadata = AlgorithmMetadata {
            name: format!("mock_{}", filename),
            version: "1.0.0".to_string(),
            description: format!("Mock plugin from {}", path.display()),
            author: "LoneBoth AI".to_string(),
            license: "MIT".to_string(),
            api_version: self.api_version.clone(),
            dependencies: Vec::new(),
            capabilities: vec!["transform".to_string()],
        };
        
        self.validate_plugin(&metadata)?;
        
        let algorithm = Box::new(MockDynamicAlgorithm::new(metadata.clone()));
        
        Ok(Plugin::new(metadata, algorithm, path.to_path_buf()))
    }
    
    /// Create mock plugins for directory (placeholder)
    fn create_mock_plugins_for_directory(&self, directory: &Path) -> Result<Vec<Plugin>, RuntimeError> {
        let mut plugins = Vec::new();
        
        // Create some example mock plugins
        let plugin_names = vec!["image_processor", "text_classifier", "data_transformer"];
        
        for name in plugin_names {
            let metadata = AlgorithmMetadata {
                name: name.to_string(),
                version: "1.0.0".to_string(),
                description: format!("Mock {} plugin", name),
                author: "LoneBoth AI".to_string(),
                license: "MIT".to_string(),
                api_version: self.api_version.clone(),
                dependencies: Vec::new(),
                capabilities: vec!["transform".to_string(), "process".to_string()],
            };
            
            self.validate_plugin(&metadata)?;
            
            let algorithm = Box::new(MockDynamicAlgorithm::new(metadata.clone()));
            let plugin_path = directory.join(format!("{}.so", name));
            
            plugins.push(Plugin::new(metadata, algorithm, plugin_path));
        }
        
        Ok(plugins)
    }
}

impl Default for DynamicLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock implementation of DynamicAlgorithm for prototype
pub struct MockDynamicAlgorithm {
    metadata: AlgorithmMetadata,
    initialized: bool,
}

impl MockDynamicAlgorithm {
    pub fn new(metadata: AlgorithmMetadata) -> Self {
        Self {
            metadata,
            initialized: false,
        }
    }
}

#[async_trait]
impl DynamicAlgorithm for MockDynamicAlgorithm {
    async fn initialize(&mut self) -> Result<(), RuntimeError> {
        log::debug!("Initializing mock algorithm: {}", self.metadata.name);
        
        // Simulate initialization time
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        self.initialized = true;
        Ok(())
    }
    
    async fn execute(&self, input: &[u8]) -> Result<Vec<u8>, RuntimeError> {
        if !self.initialized {
            return Err(RuntimeError::RuntimeError {
                reason: "Algorithm not initialized".to_string(),
            });
        }
        
        log::debug!("Executing mock algorithm: {} with {} bytes", 
                   self.metadata.name, input.len());
        
        // Simulate processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        
        // Simple transformation: reverse the bytes
        let output = input.iter().rev().cloned().collect();
        
        Ok(output)
    }
    
    async fn cleanup(&mut self) -> Result<(), RuntimeError> {
        log::debug!("Cleaning up mock algorithm: {}", self.metadata.name);
        self.initialized = false;
        Ok(())
    }
    
    fn get_metadata(&self) -> AlgorithmMetadata {
        self.metadata.clone()
    }
}

/// Runtime manager for coordinating dynamic algorithm execution
pub struct RuntimeManager {
    loader: DynamicLoader,
    active_plugins: HashMap<String, Arc<RwLock<Plugin>>>,
    execution_stats: HashMap<String, ExecutionStats>,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    pub total_executions: u64,
    pub total_execution_time_ms: u64,
    pub average_execution_time_ms: f64,
    pub last_execution: Option<std::time::Instant>,
    pub error_count: u64,
}

impl RuntimeManager {
    pub fn new() -> Self {
        Self {
            loader: DynamicLoader::new(),
            active_plugins: HashMap::new(),
            execution_stats: HashMap::new(),
        }
    }
    
    pub async fn initialize(&mut self) -> Result<(), RuntimeError> {
        log::info!("Initializing runtime manager");
        
        let discovered = self.loader.discover_plugins().await?;
        log::info!("Discovered {} plugins", discovered.len());
        
        // Initialize all discovered plugins
        for plugin_name in discovered {
            if let Some(plugin) = self.loader.get_plugin(&plugin_name) {
                let mut plugin_guard = plugin.write().await;
                plugin_guard.algorithm.initialize().await?;
                drop(plugin_guard);
                
                self.active_plugins.insert(plugin_name.clone(), plugin);
                self.execution_stats.insert(plugin_name, ExecutionStats::default());
            }
        }
        
        log::info!("Runtime manager initialized with {} active plugins", 
                  self.active_plugins.len());
        Ok(())
    }
    
    pub async fn execute_plugin(&mut self, name: &str, input: &[u8]) -> Result<Vec<u8>, RuntimeError> {
        let start = std::time::Instant::now();
        
        let plugin = self.active_plugins.get(name)
            .ok_or_else(|| RuntimeError::PluginNotFound { name: name.to_string() })?
            .clone();
        
        let result = {
            let mut plugin_guard = plugin.write().await;
            plugin_guard.execute(input).await
        };
        
        let execution_time = start.elapsed().as_millis() as u64;
        
        // Update statistics
        let stats = self.execution_stats.entry(name.to_string()).or_default();
        match &result {
            Ok(_) => {
                stats.total_executions += 1;
                stats.total_execution_time_ms += execution_time;
                stats.average_execution_time_ms = 
                    stats.total_execution_time_ms as f64 / stats.total_executions as f64;
                stats.last_execution = Some(start);
            }
            Err(_) => {
                stats.error_count += 1;
            }
        }
        
        result
    }
    
    pub fn get_plugin_stats(&self, name: &str) -> Option<&ExecutionStats> {
        self.execution_stats.get(name)
    }
    
    pub fn list_active_plugins(&self) -> Vec<String> {
        self.active_plugins.keys().cloned().collect()
    }
    
    pub async fn reload_plugin(&mut self, name: &str) -> Result<(), RuntimeError> {
        // Unload existing plugin
        if self.active_plugins.contains_key(name) {
            self.loader.unload_plugin(name).await?;
            self.active_plugins.remove(name);
            self.execution_stats.remove(name);
        }
        
        // Reload from original path (simplified for prototype)
        let discovered = self.loader.discover_plugins().await?;
        if discovered.contains(&name.to_string()) {
            if let Some(plugin) = self.loader.get_plugin(name) {
                let mut plugin_guard = plugin.write().await;
                plugin_guard.algorithm.initialize().await?;
                drop(plugin_guard);
                
                self.active_plugins.insert(name.to_string(), plugin);
                self.execution_stats.insert(name.to_string(), ExecutionStats::default());
            }
        }
        
        Ok(())
    }
}

impl Default for RuntimeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_algorithm_metadata() {
        let metadata = AlgorithmMetadata::default();
        assert_eq!(metadata.name, "unknown");
        assert_eq!(metadata.api_version, "1.0.0");
    }

    #[tokio::test]
    async fn test_mock_dynamic_algorithm() {
        let metadata = AlgorithmMetadata::default();
        let mut algorithm = MockDynamicAlgorithm::new(metadata);
        
        // Test initialization
        algorithm.initialize().await.unwrap();
        
        // Test execution
        let input = vec![1, 2, 3, 4];
        let output = algorithm.execute(&input).await.unwrap();
        assert_eq!(output, vec![4, 3, 2, 1]); // Reversed
        
        // Test cleanup
        algorithm.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_dynamic_loader() {
        let mut loader = DynamicLoader::new();
        
        // Test plugin discovery
        let discovered = loader.discover_plugins().await.unwrap();
        assert!(!discovered.is_empty());
        
        // Test plugin retrieval
        if let Some(plugin_name) = discovered.first() {
            let plugin = loader.get_plugin(plugin_name);
            assert!(plugin.is_some());
        }
    }

    #[tokio::test]
    async fn test_runtime_manager() {
        let mut manager = RuntimeManager::new();
        
        // Test initialization
        manager.initialize().await.unwrap();
        
        let active_plugins = manager.list_active_plugins();
        assert!(!active_plugins.is_empty());
        
        // Test plugin execution
        if let Some(plugin_name) = active_plugins.first() {
            let input = vec![1, 2, 3, 4];
            let result = manager.execute_plugin(plugin_name, &input).await;
            assert!(result.is_ok());
            
            // Check stats
            let stats = manager.get_plugin_stats(plugin_name);
            assert!(stats.is_some());
            assert_eq!(stats.unwrap().total_executions, 1);
        }
    }

    #[tokio::test]
    async fn test_dynamic_algorithm_wrapper() {
        let metadata = AlgorithmMetadata {
            name: "test_wrapper".to_string(),
            version: "1.0.0".to_string(),
            api_version: "1.0.0".to_string(),
            ..Default::default()
        };
        
        let mut mock_algorithm = MockDynamicAlgorithm::new(metadata.clone());
        mock_algorithm.initialize().await.unwrap();
        
        let plugin = Plugin::new(
            metadata,
            Box::new(mock_algorithm),
            PathBuf::from("test.so"),
        );
        
        let wrapper = DynamicAlgorithmWrapper::new(plugin);
        
        // Test algorithm execution through wrapper
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = wrapper.execute(&input).await.unwrap();
        
        assert!(!result.data.is_empty());
        assert_eq!(result.metadata.get("type").unwrap(), "dynamic");
    }
}