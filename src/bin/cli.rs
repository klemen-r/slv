//! SLV Options CLI
//!
//! Command-line interface for the SLV options pricing system.

use slv_options::prelude::*;
use slv_options::models::black_scholes;

fn main() {
    println!("SLV Options Pricing System");
    println!("==========================\n");

    // Example: Black-Scholes pricing
    let spot = 500.0;
    let strike = 500.0; // ATM
    let time = 30.0 / 365.0; // 30 days
    let rate = 0.05;
    let div = 0.0;
    let vol = 0.20;

    println!("Black-Scholes Pricing Example:");
    println!("  Spot: ${:.2}", spot);
    println!("  Strike: ${:.2}", strike);
    println!("  Time: {:.0} days", time * 365.0);
    println!("  Rate: {:.1}%", rate * 100.0);
    println!("  Vol: {:.1}%\n", vol * 100.0);

    let call_price = black_scholes::price(spot, strike, rate, div, vol, time, OptionType::Call);
    let put_price = black_scholes::price(spot, strike, rate, div, vol, time, OptionType::Put);

    println!("Option Prices:");
    println!("  Call: ${:.2}", call_price);
    println!("  Put: ${:.2}", put_price);

    // Compute Greeks
    let greeks = black_scholes::greeks(spot, strike, rate, div, vol, time, OptionType::Call);
    println!("\nCall Greeks:");
    println!("  Delta: {:.4}", greeks.delta);
    println!("  Gamma: {:.6}", greeks.gamma);
    println!("  Theta: {:.4}", greeks.theta);
    println!("  Vega: {:.4}", greeks.vega);
    println!("  Rho: {:.4}", greeks.rho);

    // Test IV solver
    println!("\nImplied Volatility Solver:");
    let market_price = call_price;
    match black_scholes::implied_volatility(market_price, spot, strike, rate, div, time, OptionType::Call) {
        Ok(iv) => println!("  Recovered IV: {:.2}% (expected: {:.2}%)", iv * 100.0, vol * 100.0),
        Err(e) => println!("  IV solve failed: {:?}", e),
    }

    // Try fetching real data
    println!("\n--- Live Data ---");
    println!("Attempting to fetch QQQ options from Yahoo Finance...\n");

    let yahoo = YahooClient::new();

    match yahoo.get_quote("QQQ") {
        Ok(quote) => {
            println!("QQQ Quote:");
            println!("  Price: ${:.2}", quote.price);
            println!("  Bid: ${:.2}", quote.bid.unwrap_or(0.0));
            println!("  Ask: ${:.2}", quote.ask.unwrap_or(0.0));

            // Try to get available expirations
            match yahoo.get_expirations("QQQ") {
                Ok(exps) => {
                    println!("  Expirations: {} available", exps.len());
                    if let Some(exp) = exps.first() {
                        println!("  First expiry: {}", exp);
                    }
                }
                Err(e) => println!("  Could not fetch expirations: {:?}", e),
            }
        }
        Err(e) => {
            println!("Could not fetch QQQ: {:?}", e);
            println!("(This is expected if you're offline or Yahoo API is unavailable)");
        }
    }

    println!("\n--- Done ---");
}
