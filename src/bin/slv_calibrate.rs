//! SLV Calibration from JSON options data
//!
//! Loads options data fetched by Python script and calibrates the SLV model.

use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};
use chrono::NaiveDate;
use ndarray::Array2;

use slv_options::prelude::*;
use slv_options::models::black_scholes;

#[derive(Debug, Deserialize)]
struct OptionsData {
    symbol: String,
    spot: f64,
    timestamp: String,
    chains: Vec<ChainData>,
}

#[derive(Debug, Deserialize)]
struct ChainData {
    expiry: String,
    calls: Vec<OptionData>,
    puts: Vec<OptionData>,
}

#[derive(Debug, Deserialize)]
struct OptionData {
    strike: f64,
    bid: Option<f64>,
    ask: Option<f64>,
    last: Option<f64>,
    iv: Option<f64>,
    volume: i64,
    open_interest: i64,
}

fn main() {
    println!("SLV Model Calibration");
    println!("=====================\n");

    // Load options data
    let data_path = "D:/Stochastic local volatility/slv-options/data/spy_options.json";

    let file = match File::open(data_path) {
        Ok(f) => f,
        Err(e) => {
            println!("Error: Could not open {}: {}", data_path, e);
            println!("Run: python scripts/fetch_options.py SPY");
            return;
        }
    };

    let reader = BufReader::new(file);
    let data: OptionsData = match serde_json::from_reader(reader) {
        Ok(d) => d,
        Err(e) => {
            println!("Error parsing JSON: {}", e);
            return;
        }
    };

    println!("Loaded {} options data", data.symbol);
    println!("  Spot: ${:.2}", data.spot);
    println!("  Timestamp: {}", data.timestamp);
    println!("  Chains: {}", data.chains.len());

    let spot = data.spot;
    let rate = 0.045; // ~4.5% risk-free rate
    let div = 0.013;  // ~1.3% dividend yield for SPY

    // Build IV surface from market data
    println!("\n--- Building IV Surface ---\n");

    let mut strikes: Vec<f64> = Vec::new();
    let mut times: Vec<f64> = Vec::new();
    let mut expiries: Vec<NaiveDate> = Vec::new();

    let today = chrono::Utc::now().date_naive();

    // Collect all unique strikes and compute times
    for chain in &data.chains {
        let expiry = NaiveDate::parse_from_str(&chain.expiry, "%Y-%m-%d").unwrap();
        let tte = (expiry - today).num_days() as f64 / 365.25;

        if tte <= 0.0 {
            continue; // Skip expired
        }

        times.push(tte);
        expiries.push(expiry);

        for call in &chain.calls {
            if !strikes.contains(&call.strike) {
                strikes.push(call.strike);
            }
        }
    }

    strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Filter strikes to reasonable range around ATM
    let atm_idx = strikes.iter().position(|&k| k >= spot).unwrap_or(strikes.len() / 2);
    let start = atm_idx.saturating_sub(15);
    let end = (atm_idx + 15).min(strikes.len());
    strikes = strikes[start..end].to_vec();

    println!("Surface dimensions:");
    println!("  Strikes: {} ({:.0} to {:.0})", strikes.len(), strikes[0], strikes[strikes.len()-1]);
    println!("  Expiries: {} ({:.0} to {:.0} days)", times.len(), times[0] * 365.0, times[times.len()-1] * 365.0);

    // Build IV grid
    let mut iv_grid: Array2<f64> = Array2::zeros((strikes.len(), times.len()));
    let mut valid_count = 0;

    for (ti, chain) in data.chains.iter().enumerate() {
        let expiry = NaiveDate::parse_from_str(&chain.expiry, "%Y-%m-%d").unwrap();
        let tte = (expiry - today).num_days() as f64 / 365.25;

        if tte <= 0.0 {
            continue;
        }

        for call in &chain.calls {
            if let Some(si) = strikes.iter().position(|&k| (k - call.strike).abs() < 0.01) {
                // Try to compute IV from mid price
                let mid = match (call.bid, call.ask) {
                    (Some(b), Some(a)) if b > 0.0 && a > 0.0 => (b + a) / 2.0,
                    _ => call.last.unwrap_or(0.0),
                };

                if mid > 0.01 {
                    // Use market IV if available, otherwise compute
                    let iv = if let Some(market_iv) = call.iv {
                        if market_iv > 0.01 && market_iv < 3.0 {
                            market_iv
                        } else {
                            continue;
                        }
                    } else {
                        match black_scholes::implied_volatility(mid, spot, call.strike, rate, div, tte, OptionType::Call) {
                            Ok(iv) if iv > 0.01 && iv < 3.0 => iv,
                            _ => continue,
                        }
                    };

                    iv_grid[[si, ti]] = iv;
                    valid_count += 1;
                }
            }
        }
    }

    println!("  Valid IV points: {}", valid_count);

    // Fill any gaps with interpolation (simple linear for now)
    for ti in 0..times.len() {
        let mut last_valid = None;
        for si in 0..strikes.len() {
            if iv_grid[[si, ti]] > 0.0 {
                last_valid = Some((si, iv_grid[[si, ti]]));
            } else if let Some((last_si, last_iv)) = last_valid {
                // Forward fill
                iv_grid[[si, ti]] = last_iv;
            }
        }
        // Backward fill
        let mut last_valid = None;
        for si in (0..strikes.len()).rev() {
            if iv_grid[[si, ti]] > 0.0 {
                last_valid = Some(iv_grid[[si, ti]]);
            } else if let Some(last_iv) = last_valid {
                iv_grid[[si, ti]] = last_iv;
            }
        }
    }

    // Create VolSurface
    let vol_surface = VolSurface::from_grid(
        &data.symbol,
        spot,
        today,
        strikes.clone(),
        times.clone(),
        expiries.clone(),
        iv_grid.clone(),
        rate,
        div,
    );

    // Print IV smile for each expiry
    println!("\n--- Implied Volatility Surface ---\n");

    print!("Strike\\Days |");
    for t in &times {
        print!(" {:>5.0}", t * 365.0);
    }
    println!();
    print!("-----------+");
    for _ in &times {
        print!("------");
    }
    println!();

    for (si, &k) in strikes.iter().enumerate() {
        let moneyness = k / spot;
        let marker = if (moneyness - 1.0).abs() < 0.01 { "*" } else { " " };
        print!("{}{:>9.0} |", marker, k);
        for ti in 0..times.len() {
            let iv = iv_grid[[si, ti]];
            if iv > 0.0 {
                print!(" {:>5.1}", iv * 100.0);
            } else {
                print!("     -");
            }
        }
        println!();
    }
    println!("\n* = ATM strike");

    // Build Local Volatility Surface
    println!("\n--- Local Volatility (Dupire) ---\n");

    match LocalVolSurface::from_implied_vol(&vol_surface) {
        Ok(local_vol) => {
            println!("Local vol surface built successfully");

            // Sample local vol at a few points
            let test_times = [0.02, 0.05, 0.1];
            let test_moneyness = [0.95, 0.97, 1.0, 1.03, 1.05];

            print!("Moneyness\\T |");
            for t in test_times {
                print!(" {:>5.0}d", t * 365.0);
            }
            println!();
            print!("-----------+");
            for _ in test_times {
                print!("------");
            }
            println!();

            for m in test_moneyness {
                let k = spot * m;
                print!("  {:.0}%    |", m * 100.0);
                for t in test_times {
                    let lv = local_vol.local_vol(k, t);
                    print!(" {:>5.1}", lv * 100.0);
                }
                println!();
            }
        }
        Err(e) => {
            println!("Could not build local vol: {:?}", e);
        }
    }

    // Calibrate Heston model
    println!("\n--- Heston Model Calibration ---\n");

    // Use typical equity parameters as starting point
    let heston_params = HestonParams::typical_equity();
    let heston_model = HestonModel::new(heston_params.clone(), rate, div);

    println!("Heston parameters (typical equity):");
    println!("  v0 (initial var): {:.4}", heston_params.v0);
    println!("  kappa (mean rev): {:.4}", heston_params.kappa);
    println!("  theta (long var): {:.4}", heston_params.theta);
    println!("  sigma (vol of vol): {:.4}", heston_params.sigma);
    println!("  rho (correlation): {:.4}", heston_params.rho);

    // Price ATM options with Heston vs market
    println!("\nHeston vs Market prices (ATM):");
    let atm_strike = strikes.iter()
        .min_by(|a, b| ((*a - spot).abs()).partial_cmp(&(*b - spot).abs()).unwrap())
        .copied()
        .unwrap_or(spot);

    for (ti, &t) in times.iter().enumerate().take(4) {
        let market_iv = iv_grid[[strikes.iter().position(|&k| (k - atm_strike).abs() < 0.1).unwrap_or(0), ti]];
        let market_price = black_scholes::price(spot, atm_strike, rate, div, market_iv, t, OptionType::Call);
        let heston_price = heston_model.price(spot, atm_strike, t, OptionType::Call);

        println!("  {:.0}d: Market ${:.2} (IV={:.1}%), Heston ${:.2}",
            t * 365.0, market_price, market_iv * 100.0, heston_price);
    }

    // Build SLV Model
    println!("\n--- SLV Model ---\n");

    let slv_params = SLVParams::typical_equity(0.5); // 50% mixing

    match SLVModel::from_iv_surface(&vol_surface, slv_params.clone()) {
        Ok(slv_model) => {
            println!("SLV model calibrated successfully");
            println!("  Mixing parameter (alpha): {:.1}%", slv_params.alpha * 100.0);

            // Price some options
            println!("\nSLV Monte Carlo pricing (10k paths):");

            let test_strikes = [spot * 0.95, spot, spot * 1.05];
            let test_time = times[2.min(times.len()-1)];

            for &k in &test_strikes {
                let mc_price = slv_model.mc_price(k, test_time, OptionType::Call, 10000, 50);
                let si = strikes.iter().position(|&s| (s - k).abs() < 1.0).unwrap_or(0);
                let ti = 2.min(times.len()-1);
                let market_iv = iv_grid[[si, ti]];
                let bs_price = black_scholes::price(spot, k, rate, div, market_iv, test_time, OptionType::Call);

                println!("  K={:.0}: SLV=${:.2}, BS=${:.2}, diff={:.2}%",
                    k, mc_price, bs_price, (mc_price - bs_price) / bs_price * 100.0);
            }
        }
        Err(e) => {
            println!("Could not build SLV model: {:?}", e);
        }
    }

    println!("\n--- Calibration Complete ---");
}
