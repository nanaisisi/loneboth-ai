use loneboth_ai::{LonebothAI, Config, CoordinationMode, AlgorithmType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loneboth AI Framework Demo");
    println!("==========================");

    // Create AI instance with default configuration
    let ai = LonebothAI::new();
    println!("Created AI instance with default configuration");

    // Test with sample data
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("Input data: {:?}", input_data);

    // Process data
    let result = ai.process(&input_data)?;
    println!("Processing result: {:?}", result);

    // Test with different configurations
    println!("\nTesting different configurations:");
    
    // Static algorithm with group coordination
    let config = Config {
        gpu_enabled: true,
        coordination_mode: CoordinationMode::Group,
        verification_enabled: true,
        algorithm_type: AlgorithmType::Static,
    };
    
    let ai_group = LonebothAI::with_config(config);
    let group_result = ai_group.process(&input_data)?;
    println!("Group coordination result: {:?}", group_result);

    // Dynamic algorithm with individual coordination
    let config = Config {
        gpu_enabled: false,
        coordination_mode: CoordinationMode::Individual,
        verification_enabled: true,
        algorithm_type: AlgorithmType::Dynamic,
    };
    
    let ai_dynamic = LonebothAI::with_config(config);
    let dynamic_result = ai_dynamic.process(&input_data)?;
    println!("Dynamic algorithm result: {:?}", dynamic_result);

    // Hybrid coordination
    let config = Config {
        gpu_enabled: true,
        coordination_mode: CoordinationMode::Hybrid,
        verification_enabled: true,
        algorithm_type: AlgorithmType::Static,
    };
    
    let ai_hybrid = LonebothAI::with_config(config);
    let hybrid_result = ai_hybrid.process(&input_data)?;
    println!("Hybrid coordination result: {:?}", hybrid_result);

    println!("\nDemo completed successfully!");
    Ok(())
}
