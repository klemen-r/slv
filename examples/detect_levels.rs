//! Example: Detect levels from a volatility surface
//!
//! Run with: cargo run --example detect_levels

use ndarray::Array2;
use slv_options::prelude::*;

fn main() {
    // Create sample data: strikes around ATM
    let spot = 500.0;
    let strikes: Vec<f64> = (0..21).map(|i| 490.0 + i as f64).collect();
    let times = vec![0.003, 0.008, 0.014]; // ~1d, ~3d, ~5d

    // Build a sample IV grid with some anomalies
    let n_strikes = strikes.len();
    let n_times = times.len();
    let mut iv_grid: Array2<f64> = Array2::from_elem((n_strikes, n_times), 0.20); // Base 20% vol

    // Add a spike (elevated vol) at strike 500
    let spike_idx = strikes.iter().position(|&k| k == 500.0).unwrap();
    for ti in 0..n_times {
        iv_grid[[spike_idx, ti]] = 0.28; // 28% vol - spike
    }

    // Add an air pocket (depressed vol) at strike 505
    let pocket_idx = strikes.iter().position(|&k| k == 505.0).unwrap();
    for ti in 0..n_times {
        iv_grid[[pocket_idx, ti]] = 0.14; // 14% vol - air pocket
    }

    // Convert IV to local variance (simplified: var = iv^2)
    let mut local_var: Array2<f64> = Array2::zeros((n_strikes, n_times));
    for si in 0..n_strikes {
        for ti in 0..n_times {
            let iv: f64 = iv_grid[[si, ti]];
            local_var[[si, ti]] = iv * iv;
        }
    }

    // Configure level detection
    let config = LevelConfig {
        dislocation: DislocationConfig {
            threshold: 2.0, // Lower threshold for this example
            ..Default::default()
        },
        confirmation: ConfirmationConfig {
            include_low_confidence: true,
            medium_confidence_threshold: 0.5,
            ..Default::default()
        },
        ..Default::default()
    };

    // Detect levels
    let detector = LevelDetector::with_config(config);
    let result = detector.detect_from_grid(&strikes, &times, &local_var, spot);

    // Print results
    println!("=== Level Detection Results ===\n");
    println!("Spot: {:.2}", result.spot);
    println!("Strike spacing: {:.2}", result.strike_spacing);
    println!("Total levels: {}\n", result.levels.len());

    println!("--- Detected Levels ---\n");
    for level in &result.levels {
        println!(
            "Strike {:.0}: {} | {:?} | z={:.2} | dist={:+.1} strikes | {}",
            level.strike,
            level.kind.short_label(),
            level.confidence,
            level.max_z_score(),
            level.distance_strikes,
            if level.priority { "PRIORITY" } else { "far" }
        );
    }

    println!("\n--- Summary ---\n");
    println!("Spikes (Walls/Pivots): {}", result.spikes().len());
    println!("Air Pockets (Acceleration): {}", result.air_pockets().len());
    println!("High confidence: {}", result.high_confidence().len());
    println!("Priority (within band): {}", result.priority_levels().len());

    // Show tooltip for first level
    if let Some(level) = result.levels.first() {
        println!("\n--- Example Tooltip ---\n");
        println!("{}", level.tooltip());
    }
}
