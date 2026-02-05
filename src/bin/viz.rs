//! SLV Options Visualization
//!
//! Visualization tool for volatility surfaces and model outputs.

fn main() {
    println!("SLV Options Visualization");
    println!("=========================\n");

    // Create a sample IV surface for demonstration
    println!("Creating sample volatility surface...\n");

    let spot: f64 = 500.0;
    let strikes: Vec<f64> = (80..=120).map(|p| spot * (p as f64 / 100.0)).collect();
    let maturities: Vec<f64> = vec![0.0833, 0.1667, 0.25, 0.5, 1.0]; // 1m, 2m, 3m, 6m, 1y

    // Generate sample vols with smile
    let atm_vol: f64 = 0.20;
    let skew: f64 = -0.1;
    let smile: f64 = 0.05;

    println!("Sample Surface Parameters:");
    println!("  Spot: ${:.2}", spot);
    println!("  ATM Vol: {:.1}%", atm_vol * 100.0);
    println!("  Skew: {:.2}", skew);
    println!("  Smile: {:.2}", smile);
    println!();

    // Print vol surface as ASCII grid
    println!("Implied Volatility Surface (%):");
    println!("Strike\\TTM |  1M    2M    3M    6M    1Y");
    println!("-----------+--------------------------------");

    for &k in strikes.iter().step_by(5) {
        let moneyness = (k / spot).ln();
        print!("  {:>6.0}   |", k);

        for &t in &maturities {
            // SSVI-like parameterization
            let total_var = atm_vol.powi(2) * t;
            let vol = (total_var + skew * moneyness * t.sqrt() + smile * moneyness.powi(2)).sqrt() / t.sqrt();
            print!(" {:>5.1}", vol * 100.0);
        }
        println!();
    }

    println!();
    println!("Local Volatility Surface (Dupire):");
    println!("Strike\\TTM |  1M    2M    3M    6M    1Y");
    println!("-----------+--------------------------------");

    for &k in strikes.iter().step_by(5) {
        let moneyness = (k / spot).ln();
        print!("  {:>6.0}   |", k);

        for &t in &maturities {
            // Approximate local vol (simplified)
            let iv = atm_vol + skew * moneyness / t.sqrt() + smile * moneyness.powi(2) / t;
            let local_vol = iv * (1.0 + 0.5 * moneyness.powi(2) / t).sqrt();
            print!(" {:>5.1}", local_vol.min(1.0) * 100.0);
        }
        println!();
    }

    println!("\n--- Visualization complete ---");
    println!("Future versions will include:");
    println!("  - Interactive 3D surface plots");
    println!("  - Real-time vol surface updates");
    println!("  - Greeks heatmaps");
    println!("  - Model calibration diagnostics");
}
