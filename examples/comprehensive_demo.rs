/// Example demonstrating the LoneBoth AI Framework capabilities
use std::error::Error;
use loneboth_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the framework
    let _framework = LoneBothAI::new()?;
    
    println!("🧠 LoneBoth AI Framework Example");
    println!("=================================\n");
    
    // Example 1: Individual Algorithm Coordination
    println!("📋 Example 1: Individual Algorithm Execution");
    println!("-------------------------------------------");
    
    let mut individual_coordinator = IndividualCoordinator::new();
    
    // Add a simple algorithm
    let simple_algorithm = StaticAlgorithm::new(
        "vector_multiply".to_string(),
        "1.0.0".to_string(),
        |input| input.iter().map(|x| x * 3.0).collect(),
    );
    
    individual_coordinator.add_algorithm("vector_multiply".to_string(), Box::new(simple_algorithm));
    individual_coordinator.set_current_algorithm("vector_multiply".to_string())?;
    
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("Input: {:?}", input_data);
    
    let result = individual_coordinator.execute(&input_data).await?;
    println!("Output: {:?}", result.results.get("vector_multiply").unwrap().data);
    println!("Execution time: {}ms", result.execution_time_ms);
    println!("Consensus score: {}\n", result.consensus_score);
    
    // Example 2: Group Algorithm Coordination
    println!("👥 Example 2: Group Algorithm Coordination");
    println!("------------------------------------------");
    
    let mut group_coordinator = GroupCoordinator::new()
        .with_consensus_threshold(0.7);
    
    // Add multiple algorithms
    let algorithm1 = StaticAlgorithm::new(
        "linear_transform".to_string(),
        "1.0.0".to_string(),
        |input| input.iter().map(|x| x * 2.0 + 1.0).collect(),
    );
    
    let algorithm2 = StaticAlgorithm::new(
        "quadratic_transform".to_string(),
        "1.0.0".to_string(),
        |input| input.iter().map(|x| x * x).collect(),
    );
    
    group_coordinator.add_algorithm("linear".to_string(), Box::new(algorithm1));
    group_coordinator.add_algorithm("quadratic".to_string(), Box::new(algorithm2));
    
    let group_input = vec![1.0, 2.0, 3.0];
    println!("Input: {:?}", group_input);
    
    match group_coordinator.execute_parallel(&group_input).await {
        Ok(group_result) => {
            println!("Group execution successful!");
            for (name, result) in &group_result.results {
                println!("  {}: {:?}", name, result.data);
            }
            println!("Consensus score: {}", group_result.consensus_score);
        }
        Err(e) => {
            println!("Group execution failed: {}", e);
        }
    }
    println!();
    
    // Example 3: GPU Acceleration
    println!("🚀 Example 3: GPU Acceleration");
    println!("-------------------------------");
    
    let mut gpu_accelerator = GpuAccelerator::new().await?;
    
    println!("Available GPU providers:");
    for provider in gpu_accelerator.get_available_providers() {
        println!("  - {} (GPU: {})", provider.get_name(), provider.is_gpu_accelerated());
    }
    
    // Create an executor
    gpu_accelerator.create_executor("demo".to_string(), None).await?;
    gpu_accelerator.load_model("demo", "demo_model.onnx").await?;
    
    // Create a tensor for GPU operations
    let tensor = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2]
    )?;
    
    println!("Input tensor: shape {:?}, data {:?}", tensor.shape, tensor.data);
    
    let output_tensor = gpu_accelerator.run_inference("demo", &tensor).await?;
    println!("Output tensor: shape {:?}, data {:?}", output_tensor.shape, output_tensor.data);
    println!();
    
    // Example 4: Dynamic Algorithm Loading
    println!("🔄 Example 4: Dynamic Algorithm Loading");
    println!("---------------------------------------");
    
    let mut runtime_manager = RuntimeManager::new();
    runtime_manager.initialize().await?;
    
    let active_plugins = runtime_manager.list_active_plugins();
    println!("Active plugins: {:?}", active_plugins);
    
    if let Some(plugin_name) = active_plugins.first() {
        let test_data = vec![1, 2, 3, 4, 5];
        println!("Testing plugin '{}' with data: {:?}", plugin_name, test_data);
        
        match runtime_manager.execute_plugin(plugin_name, &test_data).await {
            Ok(result) => {
                println!("Plugin result: {:?}", result);
                
                if let Some(stats) = runtime_manager.get_plugin_stats(plugin_name) {
                    println!("Execution stats: {} total executions", stats.total_executions);
                }
            }
            Err(e) => {
                println!("Plugin execution failed: {}", e);
            }
        }
    }
    println!();
    
    // Example 5: Verification System
    println!("✅ Example 5: Verification System");
    println!("---------------------------------");
    
    let mut verification_system = VerificationSystem::default();
    
    // Create some test data
    let test_input = vec![1.0, 2.0, 3.0, 4.0];
    let test_output = AlgorithmResult {
        data: vec![2.0, 4.0, 6.0, 8.0],
        metadata: std::collections::HashMap::new(),
        execution_time_ms: 5,
    };
    
    println!("Verifying algorithm output...");
    let verification_result = verification_system
        .verify_all("test_algorithm", &test_input, &test_output)
        .await?;
    
    println!("Verification passed: {}", verification_result.passed);
    println!("Integrity score: {:.2}", verification_result.integrity_score);
    println!("Checks performed: {:?}", verification_result.checks_performed);
    
    if !verification_result.warnings.is_empty() {
        println!("Warnings: {:?}", verification_result.warnings);
    }
    
    if !verification_result.errors.is_empty() {
        println!("Errors: {:?}", verification_result.errors);
    }
    
    println!("\n🎉 All examples completed successfully!");
    println!("The LoneBoth AI framework is ready for advanced AI workloads.");
    
    Ok(())
}