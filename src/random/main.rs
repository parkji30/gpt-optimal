#![recursion_limit = "256"]

mod burn_mlp;
mod custom_mlp;

fn main() {
    // Run custom implementation
    custom_mlp::train_custom_mlp();
    
    // skip line.
    println!();
    
    // Run Burn implementation
    burn_mlp::train_burn_mlp();
}
