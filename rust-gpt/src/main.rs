#![recursion_limit = "256"]

mod burn_mlp;
mod custom_mlp;

fn main() {
    println!("=== Comparing MLP Implementations ===\n");
    
    // Run custom implementation
    println!("🔧 CUSTOM IMPLEMENTATION");
    println!("========================");
    custom_mlp::train_custom_mlp();
    
    println!("\n{}\n", "=".repeat(50));
    
    // Run Burn implementation
    println!("🔥 BURN FRAMEWORK IMPLEMENTATION");
    println!("=================================");
    burn_mlp::train_burn_mlp();
}
