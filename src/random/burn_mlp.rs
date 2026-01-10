use burn::{
    backend::{Autodiff, Wgpu},
    module::Module,
    nn::{Linear, LinearConfig, loss::MseLoss},
    optim::{AdamConfig, Optimizer, GradientsParams},
    tensor::{backend::Backend, TensorData, Tensor},
};

// Define a simple MLP model
#[derive(Module, Debug)]
pub struct SimpleMLP<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    layer3: Linear<B>,
}

impl<B: Backend> SimpleMLP<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            layer1: LinearConfig::new(2, 8).init(device),   // Input: 2, Hidden: 8
            layer2: LinearConfig::new(8, 4).init(device),   // Hidden: 8, Hidden: 4
            layer3: LinearConfig::new(4, 1).init(device),   // Hidden: 4, Output: 1
        }
    }

    // Forward pass
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(input);
        let x = burn::tensor::activation::relu(x);
        let x = self.layer2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.layer3.forward(x);
        burn::tensor::activation::sigmoid(x)
    }
}

// Training function
pub fn train_burn_mlp() {
    println!("Training simple MLP with Burn framework on XOR problem...\n");
    
    // Create device
    let device = Default::default();
    
    // Create model with autodiff support
    let mut model: SimpleMLP<Autodiff<Wgpu>> = SimpleMLP::new(&device);
    
    // Create optimizer
    let mut optimizer = AdamConfig::new().init();
    
    // XOR training data
    let inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];
    let targets = [0.0f32, 1.0f32, 1.0f32, 0.0f32];
    
    println!("Starting training...");
    
    // Training loop
    for epoch in 0..1000 {
        let mut total_loss = 0.0;
        
        for (i, input_data) in inputs.iter().enumerate() {
            // Convert input to tensor (create as 2D directly)
            let input: Tensor<Autodiff<Wgpu>, 2> = Tensor::<Autodiff<Wgpu>, 2>::from_data(
                TensorData::from([[input_data[0], input_data[1]]]), 
                &device
            );
            
            // Convert target to tensor
            let target: Tensor<Autodiff<Wgpu>, 2> = Tensor::<Autodiff<Wgpu>, 2>::from_data(
                TensorData::from([[targets[i]]]), 
                &device
            );
            
            // Forward pass
            let output = model.forward(input);
            
            // Calculate loss
            let loss = MseLoss::new().forward(output, target, burn::nn::loss::Reduction::Mean);
            
            // Backward pass
            let grads = loss.backward();
            
            // Convert gradients to the expected format
            let grads = GradientsParams::from_grads(grads, &model);
            
            // Update model parameters  
            model = optimizer.step(0.01, model, grads);
            
            total_loss += loss.into_scalar();
        }
        
        if epoch % 100 == 0 {
            println!(
                "Epoch {}: Average Loss = {:.6}",
                epoch,
                total_loss / inputs.len() as f32
            );
        }
    }
    
    println!("\nTraining completed!\n");
    
    // Test the model
    println!("Testing the trained model:");
    println!("Input -> Expected : Predicted");
    println!("============================");
    
    let mut correct = 0;
    
    for (i, input_data) in inputs.iter().enumerate() {
        let input: Tensor<Autodiff<Wgpu>, 2> = Tensor::<Autodiff<Wgpu>, 2>::from_data(
            TensorData::from([[input_data[0], input_data[1]]]), 
            &device
        );
        
        let output = model.forward(input);
        let prediction: f32 = output.into_scalar();
        let predicted_class = if prediction > 0.5 { 1.0 } else { 0.0 };
        
        println!(
            "{:?} -> {:.1} : {:.4} ({:.1})",
            input_data, targets[i], prediction, predicted_class
        );
        
        if (predicted_class - targets[i]).abs() < 0.1f32 {
            correct += 1;
        }
    }
    
    println!(
        "\nFinal Accuracy: {}/{} ({:.1}%)",
        correct,
        inputs.len(),
        (correct as f32 / inputs.len() as f32) * 100.0
    );
}