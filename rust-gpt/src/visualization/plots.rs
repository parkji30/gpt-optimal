use std::io::Write;

/// Plot loss history to terminal or file
pub fn plot_loss_history(
    train_losses: &[(usize, f64)],
    valid_losses: &[(usize, f64)],
    title: &str,
) {
    println!("\n{}", "=".repeat(60));
    println!("{:^60}", title);
    println!("{}", "=".repeat(60));
    
    if train_losses.is_empty() {
        println!("No data to plot.");
        return;
    }
    
    // Combine all losses for scaling
    let all_losses: Vec<f64> = train_losses
        .iter()
        .map(|(_, l)| *l)
        .chain(valid_losses.iter().map(|(_, l)| *l))
        .collect();
    
    let min_loss = all_losses.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_loss = all_losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_loss - min_loss;
    
    let height = 15;
    let width = 50;
    
    // Create a grid
    let mut grid = vec![vec![' '; width]; height];
    
    // Downsample train losses
    let step_size = (train_losses.len() as f64 / width as f64).ceil().max(1.0) as usize;
    let train_downsampled: Vec<(usize, f64)> = train_losses
        .chunks(step_size)
        .enumerate()
        .map(|(i, chunk)| {
            let avg = chunk.iter().map(|(_, l)| l).sum::<f64>() / chunk.len() as f64;
            (i, avg)
        })
        .collect();
    
    // Plot train losses
    for (x, loss) in train_downsampled.iter() {
        if *x >= width {
            break;
        }
        let y = if range > 0.0 {
            ((max_loss - loss) / range * (height - 1) as f64) as usize
        } else {
            height / 2
        };
        let y = y.min(height - 1);
        grid[y][*x] = '●';
    }
    
    // Plot valid losses if available
    if !valid_losses.is_empty() {
        let valid_step_size = (valid_losses.len() as f64 / width as f64).ceil().max(1.0) as usize;
        let valid_downsampled: Vec<(usize, f64)> = valid_losses
            .chunks(valid_step_size)
            .enumerate()
            .map(|(i, chunk)| {
                let avg = chunk.iter().map(|(_, l)| l).sum::<f64>() / chunk.len() as f64;
                (i, avg)
            })
            .collect();
        
        for (x, loss) in valid_downsampled.iter() {
            if *x >= width {
                break;
            }
            let y = if range > 0.0 {
                ((max_loss - loss) / range * (height - 1) as f64) as usize
            } else {
                height / 2
            };
            let y = y.min(height - 1);
            grid[y][*x] = '○';
        }
    }
    
    // Print grid with axis
    for (i, row) in grid.iter().enumerate() {
        let label = if i == 0 {
            format!("{:8.4}", max_loss)
        } else if i == height - 1 {
            format!("{:8.4}", min_loss)
        } else {
            "        ".to_string()
        };
        
        print!("{} │", label);
        for c in row {
            print!("{}", c);
        }
        println!();
    }
    
    // X-axis
    print!("         └");
    for _ in 0..width {
        print!("─");
    }
    println!();
    
    // Legend
    println!("\n● Train Loss  ○ Valid Loss");
    println!("Steps: 0 to {}", train_losses.len());
}

/// Plot learning rate schedule
pub fn plot_learning_rate(
    lr_history: &[(usize, f64)],
    title: &str,
) {
    println!("\n{}", "=".repeat(60));
    println!("{:^60}", title);
    println!("{}", "=".repeat(60));
    
    if lr_history.is_empty() {
        println!("No data to plot.");
        return;
    }
    
    let lrs: Vec<f64> = lr_history.iter().map(|(_, lr)| *lr).collect();
    let min_lr = lrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_lr = lrs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_lr - min_lr;
    
    let height = 10;
    let width = 50;
    
    let mut grid = vec![vec![' '; width]; height];
    
    let step_size = (lr_history.len() as f64 / width as f64).ceil().max(1.0) as usize;
    let downsampled: Vec<(usize, f64)> = lr_history
        .chunks(step_size)
        .enumerate()
        .map(|(i, chunk)| {
            let avg = chunk.iter().map(|(_, lr)| lr).sum::<f64>() / chunk.len() as f64;
            (i, avg)
        })
        .collect();
    
    for (x, lr) in downsampled.iter() {
        if *x >= width {
            break;
        }
        let y = if range > 0.0 {
            ((max_lr - lr) / range * (height - 1) as f64) as usize
        } else {
            height / 2
        };
        let y = y.min(height - 1);
        grid[y][*x] = '▪';
    }
    
    for (i, row) in grid.iter().enumerate() {
        let label = if i == 0 {
            format!("{:8.2e}", max_lr)
        } else if i == height - 1 {
            format!("{:8.2e}", min_lr)
        } else {
            "        ".to_string()
        };
        
        print!("{} │", label);
        for c in row {
            print!("{}", c);
        }
        println!();
    }
    
    print!("         └");
    for _ in 0..width {
        print!("─");
    }
    println!();
    println!("\nSteps: 0 to {}", lr_history.len());
}

/// Save loss history to CSV file
pub fn save_loss_csv(
    train_losses: &[(usize, f64)],
    valid_losses: &[(usize, f64)],
    path: &str,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    
    writeln!(file, "step,train_loss,valid_loss")?;
    
    let max_len = train_losses.len().max(valid_losses.len());
    
    for i in 0..max_len {
        let train = train_losses.get(i).map(|(_, l)| *l);
        let valid = valid_losses.get(i).map(|(_, l)| *l);
        
        let step = train_losses.get(i).map(|(s, _)| *s)
            .or_else(|| valid_losses.get(i).map(|(s, _)| *s))
            .unwrap_or(i);
        
        writeln!(
            file,
            "{},{},{}",
            step,
            train.map(|v| format!("{:.6}", v)).unwrap_or_default(),
            valid.map(|v| format!("{:.6}", v)).unwrap_or_default(),
        )?;
    }
    
    println!("Loss history saved to {}", path);
    Ok(())
}

