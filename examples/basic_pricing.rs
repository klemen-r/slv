//! Example: Basic options pricing with Black-Scholes
//!
//! Run with: cargo run --example basic_pricing

use slv_options::prelude::*;

fn main() {
    // Option parameters
    let spot = 500.0;
    let strike = 505.0;
    let time = 0.25; // 3 months
    let rate = 0.05; // 5% risk-free rate
    let div = 0.01; // 1% dividend yield
    let vol = 0.20; // 20% volatility

    println!("=== Black-Scholes Pricing ===\n");
    println!("Spot:     ${:.2}", spot);
    println!("Strike:   ${:.2}", strike);
    println!("Time:     {:.2} years ({:.0} days)", time, time * 365.0);
    println!("Rate:     {:.1}%", rate * 100.0);
    println!("Div:      {:.1}%", div * 100.0);
    println!("Vol:      {:.1}%\n", vol * 100.0);

    // Price call option
    let call_price = bs_price(spot, strike, time, rate, div, vol, OptionType::Call);
    println!("Call Price: ${:.4}", call_price);

    // Price put option
    let put_price = bs_price(spot, strike, time, rate, div, vol, OptionType::Put);
    println!("Put Price:  ${:.4}", put_price);

    // Verify put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
    let parity_lhs = call_price - put_price;
    let parity_rhs = spot * (-div * time).exp() - strike * (-rate * time).exp();
    println!("\nPut-Call Parity Check:");
    println!("  C - P = {:.4}", parity_lhs);
    println!("  S*e^(-qT) - K*e^(-rT) = {:.4}", parity_rhs);
    println!("  Difference: {:.6}", (parity_lhs - parity_rhs).abs());

    // Calculate Greeks for the call
    println!("\n=== Greeks (Call) ===\n");
    let greeks = bs_greeks(spot, strike, time, rate, div, vol, OptionType::Call);
    println!("Delta:  {:.4}", greeks.delta);
    println!("Gamma:  {:.4}", greeks.gamma);
    println!("Theta:  {:.4} (per day: {:.4})", greeks.theta, greeks.theta / 365.0);
    println!("Vega:   {:.4}", greeks.vega);
    println!("Rho:    {:.4}", greeks.rho);

    // Implied volatility calculation
    println!("\n=== Implied Volatility ===\n");
    let market_price = call_price + 0.50; // Simulated market price
    match implied_volatility(market_price, spot, strike, time, rate, div, OptionType::Call) {
        Ok(iv) => println!(
            "Market price ${:.4} implies vol: {:.2}%",
            market_price,
            iv * 100.0
        ),
        Err(e) => println!("Could not solve for IV: {}", e),
    }
}
