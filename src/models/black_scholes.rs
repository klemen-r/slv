//! Black-Scholes Model
//!
//! Provides:
//! - European option pricing
//! - Greeks computation
//! - Implied volatility solver (Newton-Raphson with bisection fallback)
//!
//! The Black-Scholes model serves as the baseline and is used for:
//! - Converting market prices to implied volatilities
//! - Computing model-free Greeks
//! - Benchmarking more complex models

use std::f64::consts::PI;
use statrs::distribution::{ContinuousCDF, Normal};
use crate::core::{OptionType, Greeks, SLVError, SLVResult};

/// Standard normal CDF
pub fn norm_cdf(x: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.cdf(x)
}

/// Standard normal PDF
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Black-Scholes d1 parameter
pub fn d1(spot: f64, strike: f64, rate: f64, div: f64, vol: f64, time: f64) -> f64 {
    let forward = spot * ((rate - div) * time).exp();
    ((forward / strike).ln() + 0.5 * vol * vol * time) / (vol * time.sqrt())
}

/// Black-Scholes d2 parameter
pub fn d2(spot: f64, strike: f64, rate: f64, div: f64, vol: f64, time: f64) -> f64 {
    d1(spot, strike, rate, div, vol, time) - vol * time.sqrt()
}

/// Black-Scholes European option price
pub fn price(
    spot: f64,
    strike: f64,
    rate: f64,
    div: f64,
    vol: f64,
    time: f64,
    option_type: OptionType,
) -> f64 {
    if time <= 0.0 {
        return option_type.intrinsic(spot, strike);
    }

    if vol <= 0.0 {
        // Zero vol = intrinsic value discounted
        let forward = spot * ((rate - div) * time).exp();
        let df = (-rate * time).exp();
        return df * option_type.intrinsic(forward, strike);
    }

    let d1 = d1(spot, strike, rate, div, vol, time);
    let d2 = d2(spot, strike, rate, div, vol, time);
    let df = (-rate * time).exp();
    let forward = spot * ((rate - div) * time).exp();

    match option_type {
        OptionType::Call => {
            df * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
        }
        OptionType::Put => {
            df * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
        }
    }
}

/// Black-Scholes Greeks
pub fn greeks(
    spot: f64,
    strike: f64,
    rate: f64,
    div: f64,
    vol: f64,
    time: f64,
    option_type: OptionType,
) -> Greeks {
    if time <= 0.0 || vol <= 0.0 {
        // At expiry or zero vol
        let delta = match option_type {
            OptionType::Call => if spot > strike { 1.0 } else { 0.0 },
            OptionType::Put => if spot < strike { -1.0 } else { 0.0 },
        };
        return Greeks::new(delta, 0.0, 0.0, 0.0, 0.0);
    }

    let d1 = d1(spot, strike, rate, div, vol, time);
    let d2 = d2(spot, strike, rate, div, vol, time);
    let df = (-rate * time).exp();
    let sqrt_t = time.sqrt();
    let pdf_d1 = norm_pdf(d1);
    let div_factor = (-div * time).exp();

    // Delta
    let delta = match option_type {
        OptionType::Call => div_factor * norm_cdf(d1),
        OptionType::Put => div_factor * (norm_cdf(d1) - 1.0),
    };

    // Gamma (same for call and put)
    let gamma = div_factor * pdf_d1 / (spot * vol * sqrt_t);

    // Vega (same for call and put, per 1% vol move)
    let vega = spot * div_factor * pdf_d1 * sqrt_t / 100.0;

    // Theta (per day)
    let term1 = -spot * div_factor * pdf_d1 * vol / (2.0 * sqrt_t);
    let theta = match option_type {
        OptionType::Call => {
            term1 - rate * strike * df * norm_cdf(d2) + div * spot * div_factor * norm_cdf(d1)
        }
        OptionType::Put => {
            term1 + rate * strike * df * norm_cdf(-d2) - div * spot * div_factor * norm_cdf(-d1)
        }
    };
    let theta_per_day = theta / 365.0;

    // Rho (per 1% rate move)
    let rho = match option_type {
        OptionType::Call => strike * time * df * norm_cdf(d2) / 100.0,
        OptionType::Put => -strike * time * df * norm_cdf(-d2) / 100.0,
    };

    let mut greeks = Greeks::new(delta, gamma, theta_per_day, vega, rho);

    // Vanna: d(delta)/d(vol) = d(vega)/d(spot)
    let vanna = -div_factor * pdf_d1 * d2 / vol;
    greeks.vanna = Some(vanna);

    // Volga (Vomma): d(vega)/d(vol)
    let volga = vega * d1 * d2 / vol;
    greeks.volga = Some(volga);

    greeks
}

/// Implied volatility solver using Newton-Raphson with bisection fallback
pub fn implied_volatility(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    div: f64,
    time: f64,
    option_type: OptionType,
) -> SLVResult<f64> {
    // Sanity checks
    if market_price <= 0.0 {
        return Err(SLVError::numerical("Non-positive option price"));
    }
    if time <= 0.0 {
        return Err(SLVError::numerical("Non-positive time to expiry"));
    }
    if spot <= 0.0 || strike <= 0.0 {
        return Err(SLVError::numerical("Non-positive spot or strike"));
    }

    // Check intrinsic value bounds
    let intrinsic = option_type.intrinsic(spot, strike);
    let df = (-rate * time).exp();

    if market_price < intrinsic * df * 0.99 {
        return Err(SLVError::numerical("Price below intrinsic value"));
    }

    // Initial guess using Brenner-Subrahmanyam approximation
    let forward = spot * ((rate - div) * time).exp();
    let atm_approx = market_price / (0.4 * spot * time.sqrt());
    let mut vol = atm_approx.clamp(0.01, 3.0);

    // Newton-Raphson iteration
    let max_iter = 100;
    let tol = 1e-8;

    for _ in 0..max_iter {
        let bs_price = price(spot, strike, rate, div, vol, time, option_type);
        let diff = bs_price - market_price;

        if diff.abs() < tol {
            return Ok(vol);
        }

        // Vega for Newton step
        let d1 = d1(spot, strike, rate, div, vol, time);
        let vega = spot * (-div * time).exp() * norm_pdf(d1) * time.sqrt();

        if vega.abs() < 1e-12 {
            break; // Vega too small, switch to bisection
        }

        let new_vol = vol - diff / vega;

        // Ensure vol stays positive
        if new_vol <= 0.0 || new_vol > 5.0 {
            break; // Out of bounds, switch to bisection
        }

        vol = new_vol;
    }

    // Fallback to bisection
    bisection_iv(market_price, spot, strike, rate, div, time, option_type)
}

/// Bisection method for IV (slower but more robust)
fn bisection_iv(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    div: f64,
    time: f64,
    option_type: OptionType,
) -> SLVResult<f64> {
    let mut low = 0.001;
    let mut high = 5.0;
    let tol = 1e-8;
    let max_iter = 100;

    for _ in 0..max_iter {
        let mid = (low + high) / 2.0;
        let bs_price = price(spot, strike, rate, div, mid, time, option_type);
        let diff = bs_price - market_price;

        if diff.abs() < tol {
            return Ok(mid);
        }

        if diff > 0.0 {
            high = mid;
        } else {
            low = mid;
        }

        if (high - low) < tol {
            return Ok(mid);
        }
    }

    Err(SLVError::numerical("IV solver did not converge"))
}

/// Compute IV surface from quote surface
pub fn compute_iv_surface(
    quotes: &crate::core::QuoteSurface,
) -> crate::core::VolSurface {
    use crate::core::{VolSurface, VolGridType};
    use ndarray::Array2;

    let mut all_strikes: Vec<f64> = Vec::new();
    let mut times: Vec<f64> = Vec::new();
    let mut expiries: Vec<chrono::NaiveDate> = Vec::new();

    // Collect all strikes and times
    for chain in &quotes.chains {
        times.push(chain.time_to_expiry);
        expiries.push(chain.expiry);

        for strike in chain.strikes() {
            if !all_strikes.iter().any(|&s| (s - strike).abs() < 0.01) {
                all_strikes.push(strike);
            }
        }
    }

    all_strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Create IV grid
    let mut vols = Array2::zeros((all_strikes.len(), times.len()));

    for (ti, chain) in quotes.chains.iter().enumerate() {
        let time = chain.time_to_expiry;
        let rate = chain.risk_free_rate;
        let div = chain.dividend_yield;

        for (si, &strike) in all_strikes.iter().enumerate() {
            // Try OTM option (call if strike > spot, put otherwise)
            let (quote, opt_type) = if strike >= chain.spot {
                (chain.call_at(strike), OptionType::Call)
            } else {
                (chain.put_at(strike), OptionType::Put)
            };

            if let Some(q) = quote {
                if let Some(price) = q.best_price() {
                    if let Ok(iv) = implied_volatility(
                        price, chain.spot, strike, rate, div, time, opt_type
                    ) {
                        vols[[si, ti]] = iv;
                    }
                }
            }
        }
    }

    // Fill missing values with interpolation
    fill_missing_vols(&mut vols);

    VolSurface::from_grid(
        &quotes.underlying,
        quotes.spot,
        chrono::Utc::now().date_naive(),
        all_strikes,
        times,
        expiries,
        vols,
        quotes.risk_free_rate,
        quotes.dividend_yield,
    )
}

/// Fill missing volatility values using simple interpolation
fn fill_missing_vols(vols: &mut ndarray::Array2<f64>) {
    let (n_strikes, n_times) = vols.dim();

    // For each time slice
    for ti in 0..n_times {
        let mut last_valid = None;
        let mut last_valid_idx = 0;

        // Forward pass: fill using last valid
        for si in 0..n_strikes {
            let vol = vols[[si, ti]];
            if vol > 0.01 {
                // If there's a gap, interpolate
                if let Some(lv) = last_valid {
                    if si > last_valid_idx + 1 {
                        let gap = si - last_valid_idx;
                        for gi in 1..gap {
                            let frac = gi as f64 / gap as f64;
                            vols[[last_valid_idx + gi, ti]] = lv + frac * (vol - lv);
                        }
                    }
                }
                last_valid = Some(vol);
                last_valid_idx = si;
            }
        }

        // Fill edges with nearest valid
        if let Some(lv) = last_valid {
            for si in (last_valid_idx + 1)..n_strikes {
                vols[[si, ti]] = lv;
            }
        }

        // Backward pass for leading zeros
        let mut first_valid = None;
        for si in 0..n_strikes {
            if vols[[si, ti]] > 0.01 {
                first_valid = Some((si, vols[[si, ti]]));
                break;
            }
        }
        if let Some((fi, fv)) = first_valid {
            for si in 0..fi {
                vols[[si, ti]] = fv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_cdf() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-10);
        assert!((norm_cdf(1.96) - 0.975).abs() < 0.001);
        assert!((norm_cdf(-1.96) - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_bs_price() {
        // ATM call, 20% vol, 1 year, 5% rate
        let call_price = price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);

        // Should be around 10.45 for these parameters
        assert!(call_price > 10.0 && call_price < 11.0);

        // Put-call parity check
        let put_price = price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Put);
        let forward = 100.0 * 0.05_f64.exp();
        let df = (-0.05_f64).exp();
        let parity = call_price - put_price - df * (forward - 100.0);
        assert!(parity.abs() < 0.01);
    }

    #[test]
    fn test_greeks() {
        let g = greeks(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);

        // ATM call delta should be around 0.5-0.6
        assert!(g.delta > 0.5 && g.delta < 0.7);

        // Gamma should be positive
        assert!(g.gamma > 0.0);

        // Theta should be negative (time decay)
        assert!(g.theta < 0.0);

        // Vega should be positive
        assert!(g.vega > 0.0);
    }

    #[test]
    fn test_implied_vol() {
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let div = 0.0;
        let vol = 0.25;
        let time = 0.5;

        let market_price = price(spot, strike, rate, div, vol, time, OptionType::Call);
        let iv = implied_volatility(market_price, spot, strike, rate, div, time, OptionType::Call).unwrap();

        assert!((iv - vol).abs() < 0.0001);
    }

    #[test]
    fn test_iv_otm() {
        // OTM put
        let spot = 100.0;
        let strike = 90.0;
        let rate = 0.05;
        let div = 0.01;
        let vol = 0.30;
        let time = 0.25;

        let market_price = price(spot, strike, rate, div, vol, time, OptionType::Put);
        let iv = implied_volatility(market_price, spot, strike, rate, div, time, OptionType::Put).unwrap();

        assert!((iv - vol).abs() < 0.001);
    }
}
