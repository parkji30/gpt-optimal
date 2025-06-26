use rand::Rng;
use std::f64::consts::E;

// Activation functions
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

// Neural Network structure
#[derive(Debug)]
pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Vec<Vec<f64>>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    bias_output: Vec<f64>,
    learning_rate: f64,
}

impl NeuralNetwork {
    // Initialize the neural network with random weights
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        
        // Initialize weights between input and hidden layer
        let mut weights_input_hidden = vec![vec![0.0; hidden_size]; input_size];
        for i in 0..input_size {
            for j in 0..hidden_size {
                weights_input_hidden[i][j] = rng.gen_range(-1.0..1.0);
            }
        }
        
        // Initialize weights between hidden and output layer
        let mut weights_hidden_output = vec![vec![0.0; output_size]; hidden_size];
        for i in 0..hidden_size {
            for j in 0..output_size {
                weights_hidden_output[i][j] = rng.gen_range(-1.0..1.0);
            }
        }
        
        // Initialize biases
        let mut bias_hidden = vec![0.0; hidden_size];
        let mut bias_output = vec![0.0; output_size];
        
        for i in 0..hidden_size {
            bias_hidden[i] = rng.gen_range(-1.0..1.0);
        }
        
        for i in 0..output_size {
            bias_output[i] = rng.gen_range(-1.0..1.0);
        }
        
        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
            learning_rate,
        }
    }
    
    // Forward propagation
    fn forward(&self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Calculate hidden layer activations
        let mut hidden_layer = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut sum = self.bias_hidden[j];
            for i in 0..self.input_size {
                sum += inputs[i] * self.weights_input_hidden[i][j];
            }
            hidden_layer[j] = sigmoid(sum);
        }
        
        // Calculate output layer activations
        let mut output_layer = vec![0.0; self.output_size];
        for k in 0..self.output_size {
            let mut sum = self.bias_output[k];
            for j in 0..self.hidden_size {
                sum += hidden_layer[j] * self.weights_hidden_output[j][k];
            }
            output_layer[k] = sigmoid(sum);
        }
        
        (hidden_layer, output_layer)
    }
    
    // Backpropagation training
    pub fn train(&mut self, inputs: &[f64], targets: &[f64]) -> f64 {
        // Forward pass
        let (hidden_layer, output_layer) = self.forward(inputs);
        
        // Calculate output layer errors
        let mut output_errors = vec![0.0; self.output_size];
        let mut total_error = 0.0;
        for k in 0..self.output_size {
            output_errors[k] = targets[k] - output_layer[k];
            total_error += output_errors[k] * output_errors[k];
        }
        total_error *= 0.5; // Mean squared error
        
        // Calculate output layer deltas
        let mut output_deltas = vec![0.0; self.output_size];
        for k in 0..self.output_size {
            output_deltas[k] = output_errors[k] * sigmoid_derivative(output_layer[k]);
        }
        
        // Calculate hidden layer errors
        let mut hidden_errors = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut error = 0.0;
            for k in 0..self.output_size {
                error += output_deltas[k] * self.weights_hidden_output[j][k];
            }
            hidden_errors[j] = error;
        }
        
        // Calculate hidden layer deltas
        let mut hidden_deltas = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            hidden_deltas[j] = hidden_errors[j] * sigmoid_derivative(hidden_layer[j]);
        }
        
        // Update weights between hidden and output layers
        for j in 0..self.hidden_size {
            for k in 0..self.output_size {
                self.weights_hidden_output[j][k] += self.learning_rate * output_deltas[k] * hidden_layer[j];
            }
        }
        
        // Update output layer biases
        for k in 0..self.output_size {
            self.bias_output[k] += self.learning_rate * output_deltas[k];
        }
        
        // Update weights between input and hidden layers
        for i in 0..self.input_size {
            for j in 0..self.hidden_size {
                self.weights_input_hidden[i][j] += self.learning_rate * hidden_deltas[j] * inputs[i];
            }
        }
        
        // Update hidden layer biases
        for j in 0..self.hidden_size {
            self.bias_hidden[j] += self.learning_rate * hidden_deltas[j];
        }
        
        total_error
    }
    
    // Predict using the trained network
    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let (_, output) = self.forward(inputs);
        output
    }
}

// Training function for custom MLP
pub fn train_custom_mlp() {
    println!("Training a simple MLP for XOR problem...\n");
    
    // Create a neural network: 2 inputs, 4 hidden neurons, 1 output
    let mut nn = NeuralNetwork::new(2, 4, 1, 0.5);
    
    // XOR training data
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    // Training loop for 1000 iterations
    println!("Training for 1000 iterations...");
    for epoch in 0..1000 {
        let mut total_error = 0.0;
        
        // Train on each example
        for (inputs, targets) in &training_data {
            let error = nn.train(inputs, targets);
            total_error += error;
        }
        
        // Print progress every 100 iterations
        if epoch % 100 == 0 {
            println!("Epoch {}: Average Error = {:.6}", epoch, total_error / training_data.len() as f64);
        }
    }
    
    println!("\nTraining completed!\n");
    
    // Test the trained network
    println!("Testing the trained network on XOR problem:");
    println!("Input -> Expected Output : Actual Output");
    println!("========================================");
    
    for (inputs, expected) in &training_data {
        let prediction = nn.predict(inputs);
        println!("{:?} -> {:?} : {:.4}", inputs, expected[0], prediction[0]);
    }
    
    // Calculate final accuracy
    let mut correct = 0;
    for (inputs, expected) in &training_data {
        let prediction = nn.predict(inputs);
        let predicted_class = if prediction[0] > 0.5 { 1.0 } else { 0.0 };
        if (predicted_class - expected[0]).abs() < 0.1 {
            correct += 1;
        }
    }
    
    println!("\nAccuracy: {}/{} ({:.1}%)", correct, training_data.len(), 
             (correct as f64 / training_data.len() as f64) * 100.0);
} 