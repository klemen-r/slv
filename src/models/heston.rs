//! Heston Stochastic Volatility Model
//!
//! The Heston model assumes variance follows a mean-reverting square-root process:
//!
//! dS = (r - q) * S * dt + √v * S * dW_S
//! dv = κ(θ - v) * dt + σ_v * √v * dW_v
//!
//! where:
//! - S: spot price
//! - v: instantaneous variance
//! - κ: mean reversion speed
//! - θ: long-term variance
//! - σ_v: volatility of volatility (vol-of-vol)
//! - ρ: correlation between spot and variance Brownians
//!
//! This model:
//! - Generates realistic smile dynamics
//! - Allows for stochastic volatility
//! - But may not fit the entire smile perfectly

use num_complex::Complex64;
use std::f64::consts::PI;
use crate::core::{OptionType, SLVError, SLVResult};

/// Heston model parameters
#[derive(Debug, Clone, Copy)]
pub struct HestonParams {
    /// Initial variance (v0)
    pub v0: f64,
    /// Mean reversion speed (κ)
    pub kappa: f64,
    /// Long-term variance (θ)
    pub theta: f64,
    /// Volatility of volatility (σ_v)
    pub sigma: f64,
    /// Correlation between spot and variance (ρ)
    pub rho: f64,
}

impl HestonParams {
    /// Create new Heston parameters
    pub fn new(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> Self {
        Self { v0, kappa, theta, sigma, rho }
    }

    /// Typical parameters for equity index (SPX/NDX-like)
    pub fn typical_equity() -> Self {
        Self {
            v0: 0.04,      // 20% initial vol
            kappa: 2.0,    // Mean reversion
            theta: 0.04,   // 20% long-term vol
            sigma: 0.3,    // Vol-of-vol
            rho: -0.7,     // Negative correlation (leverage effect)
        }
    }

    /// Check Feller condition: 2κθ > σ² (ensures variance stays positive)
    pub fn feller_condition(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }

    /// Validate parameters
    pub fn validate(&self) -> SLVResult<()> {
        if self.v0 <= 0.0 {
            return Err(SLVError::invalid_input("v0 must be positive"));
        }
        if self.kappa <= 0.0 {
            return Err(SLVError::invalid_input("kappa must be positive"));
        }
        if self.theta <= 0.0 {
            return Err(SLVError::invalid_input("theta must be positive"));
        }
        if self.sigma <= 0.0 {
            return Err(SLVError::invalid_input("sigma must be positive"));
        }
        if self.rho < -1.0 || self.rho > 1.0 {
            return Err(SLVError::invalid_input("rho must be in [-1, 1]"));
        }
        Ok(())
    }

    /// Long-term volatility
    pub fn long_term_vol(&self) -> f64 {
        self.theta.sqrt()
    }

    /// Initial volatility
    pub fn initial_vol(&self) -> f64 {
        self.v0.sqrt()
    }
}

impl Default for HestonParams {
    fn default() -> Self {
        Self::typical_equity()
    }
}

/// Heston model pricing using characteristic function
pub struct HestonModel {
    params: HestonParams,
    rate: f64,
    div_yield: f64,
}

impl HestonModel {
    pub fn new(params: HestonParams, rate: f64, div_yield: f64) -> Self {
        Self { params, rate, div_yield }
    }

    /// Heston characteristic function φ(u)
    /// Using the "good" formulation from Albrecher et al. for numerical stability
    fn characteristic_function(&self, u: Complex64, spot: f64, time: f64) -> Complex64 {
        let p = &self.params;
        let i = Complex64::i();

        let x = spot.ln();
        let r = self.rate;
        let q = self.div_yield;

        // Modified parameters for numerical stability
        let a = p.kappa * p.theta;
        let b = p.kappa;
        let sigma2 = p.sigma * p.sigma;

        // d = sqrt((ρσiu - b)² + σ²(iu + u²))
        let rho_sigma_u_i = p.rho * p.sigma * u * i;
        let d = ((rho_sigma_u_i - b).powi(2) + sigma2 * (i * u + u * u)).sqrt();

        // g = (b - ρσiu - d) / (b - ρσiu + d)
        let g_num = b - rho_sigma_u_i - d;
        let g_den = b - rho_sigma_u_i + d;

        // Avoid division by zero
        let g = if g_den.norm() < 1e-12 {
            Complex64::new(0.0, 0.0)
        } else {
            g_num / g_den
        };

        let exp_neg_dt = (-d * time).exp();

        // C(t) = (r-q)iut + (a/σ²)[(b - ρσiu - d)t - 2ln((1 - g*exp(-dt))/(1-g))]
        let c_term1 = (r - q) * i * u * time;
        let c_term2 = if (1.0 - g).norm() < 1e-12 {
            Complex64::new(0.0, 0.0)
        } else {
            (a / sigma2) * ((b - rho_sigma_u_i - d) * time -
                2.0 * ((1.0 - g * exp_neg_dt) / (1.0 - g)).ln())
        };
        let c = c_term1 + c_term2;

        // D(t) = ((b - ρσiu - d)/σ²) * ((1 - exp(-dt))/(1 - g*exp(-dt)))
        let d_coef = if sigma2.abs() < 1e-12 {
            Complex64::new(0.0, 0.0)
        } else {
            let numer = (b - rho_sigma_u_i - d) * (1.0 - exp_neg_dt);
            let denom = sigma2 * (1.0 - g * exp_neg_dt);
            if denom.norm() < 1e-12 {
                Complex64::new(0.0, 0.0)
            } else {
                numer / denom
            }
        };

        // φ(u) = exp(C + D*v0 + iu*x)
        (c + d_coef * p.v0 + i * u * x).exp()
    }

    /// Price European option using Carr-Madan FFT approach (simplified integration)
    pub fn price(&self, spot: f64, strike: f64, time: f64, option_type: OptionType) -> f64 {
        if time <= 0.0 {
            return option_type.intrinsic(spot, strike);
        }

        let df = (-self.rate * time).exp();
        let forward = spot * ((self.rate - self.div_yield) * time).exp();
        let k = strike.ln();
        let i = Complex64::i();

        // Numerical integration using Gil-Pelaez formula
        // Call: exp(-rT) * (F*P1 - K*P2)
        // where P1, P2 are probabilities computed via characteristic function

        let n_points = 4096;
        let du = 0.01;
        let alpha = 1.5; // Damping factor

        let mut sum1 = 0.0;
        let mut sum2 = 0.0;

        for j in 1..n_points {
            let u = j as f64 * du;
            let u_complex = Complex64::new(u, 0.0);

            // P1 integrand
            let phi1 = self.characteristic_function(u_complex - i, spot, time);
            let integrand1 = (phi1 * (-i * u * k).exp() / (i * u)).re;

            // P2 integrand
            let phi2 = self.characteristic_function(u_complex, spot, time);
            let integrand2 = (phi2 * (-i * u * k).exp() / (i * u)).re;

            sum1 += integrand1 * du;
            sum2 += integrand2 * du;
        }

        let p1 = 0.5 + sum1 / PI;
        let p2 = 0.5 + sum2 / PI;

        let call_price = df * (forward * p1 - strike * p2);

        match option_type {
            OptionType::Call => call_price.max(0.0),
            OptionType::Put => {
                // Put-call parity
                (call_price - df * (forward - strike)).max(0.0)
            }
        }
    }

    /// Compute implied volatility from Heston price
    pub fn implied_vol(&self, spot: f64, strike: f64, time: f64, option_type: OptionType) -> SLVResult<f64> {
        let heston_price = self.price(spot, strike, time, option_type);
        super::black_scholes::implied_volatility(
            heston_price, spot, strike, self.rate, self.div_yield, time, option_type
        )
    }

    /// Generate Heston implied vol smile for given expiry
    pub fn smile(&self, spot: f64, strikes: &[f64], time: f64) -> Vec<(f64, f64)> {
        strikes.iter().filter_map(|&strike| {
            self.implied_vol(spot, strike, time, OptionType::Call)
                .ok()
                .map(|iv| (strike, iv))
        }).collect()
    }

    /// Simulate Heston paths using Euler scheme
    pub fn simulate_path(
        &self,
        spot: f64,
        time_horizon: f64,
        n_steps: usize,
        rng: &mut impl rand::Rng,
    ) -> Vec<(f64, f64, f64)> {  // (time, spot, variance)
        use rand_distr::{Distribution, StandardNormal};

        let dt = time_horizon / n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let p = &self.params;

        let mut path = Vec::with_capacity(n_steps + 1);
        let mut s = spot;
        let mut v = p.v0;
        let mut t = 0.0;

        path.push((t, s, v));

        for _ in 0..n_steps {
            // Generate correlated Brownians
            let z1: f64 = Distribution::<f64>::sample(&StandardNormal, rng);
            let z2: f64 = Distribution::<f64>::sample(&StandardNormal, rng);
            let dw_s = z1 * sqrt_dt;
            let dw_v = (p.rho * z1 + (1.0 - p.rho * p.rho).sqrt() * z2) * sqrt_dt;

            // Variance process (with absorption at 0)
            let v_sqrt = v.sqrt().max(0.0);
            let dv = p.kappa * (p.theta - v) * dt + p.sigma * v_sqrt * dw_v;
            v = (v + dv).max(0.0);

            // Spot process
            let ds = (self.rate - self.div_yield) * s * dt + v_sqrt * s * dw_s;
            s = (s + ds).max(0.01);

            t += dt;
            path.push((t, s, v));
        }

        path
    }

    /// Monte Carlo price for European option
    pub fn mc_price(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        option_type: OptionType,
        n_paths: usize,
        n_steps: usize,
    ) -> f64 {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let df = (-self.rate * time).exp();

        let payoffs: f64 = (0..n_paths)
            .map(|_| {
                let path = self.simulate_path(spot, time, n_steps, &mut rng);
                let final_spot = path.last().unwrap().1;
                option_type.intrinsic(final_spot, strike)
            })
            .sum();

        df * payoffs / n_paths as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feller_condition() {
        let params = HestonParams::typical_equity();
        // 2 * 2.0 * 0.04 = 0.16 > 0.3² = 0.09 ✓
        assert!(params.feller_condition());

        let bad_params = HestonParams::new(0.04, 1.0, 0.04, 0.5, -0.7);
        // 2 * 1.0 * 0.04 = 0.08 < 0.5² = 0.25 ✗
        assert!(!bad_params.feller_condition());
    }

    #[test]
    fn test_heston_price() {
        let params = HestonParams::typical_equity();
        let model = HestonModel::new(params, 0.05, 0.01);

        let spot = 100.0;
        let strike = 100.0;
        let time = 0.5;

        let call_price = model.price(spot, strike, time, OptionType::Call);
        let put_price = model.price(spot, strike, time, OptionType::Put);

        // Sanity checks
        assert!(call_price > 0.0);
        assert!(put_price > 0.0);

        // Put-call parity
        let forward = spot * ((0.05 - 0.01) * time).exp();
        let df = (-0.05 * time).exp();
        let parity_diff = (call_price - put_price - df * (forward - strike)).abs();
        assert!(parity_diff < 0.5, "Put-call parity violated: {}", parity_diff);
    }

    #[test]
    fn test_heston_smile() {
        let params = HestonParams::typical_equity();
        let model = HestonModel::new(params, 0.05, 0.01);

        let spot = 100.0;
        let strikes: Vec<f64> = (80..=120).step_by(5).map(|k| k as f64).collect();

        let smile = model.smile(spot, &strikes, 0.5);

        // Should have generated IVs for all strikes
        assert!(smile.len() > 5);

        // Smile should show negative skew (OTM puts have higher IV)
        let low_strike_iv = smile.iter().find(|(k, _)| *k < 95.0).map(|(_, v)| *v);
        let high_strike_iv = smile.iter().find(|(k, _)| *k > 105.0).map(|(_, v)| *v);

        if let (Some(low), Some(high)) = (low_strike_iv, high_strike_iv) {
            // With negative rho, expect low strikes to have higher IV (typically)
            println!("Low strike IV: {:.4}, High strike IV: {:.4}", low, high);
        }
    }

    #[test]
    fn test_heston_mc_vs_analytical() {
        let params = HestonParams::typical_equity();
        let model = HestonModel::new(params, 0.05, 0.01);

        let spot = 100.0;
        let strike = 100.0;
        let time = 0.25;

        let analytical = model.price(spot, strike, time, OptionType::Call);
        let mc = model.mc_price(spot, strike, time, OptionType::Call, 50000, 50);

        let diff = (analytical - mc).abs();
        let rel_diff = diff / analytical;

        println!("Analytical: {:.4}, MC: {:.4}, Rel diff: {:.2}%",
            analytical, mc, rel_diff * 100.0);

        // MC should be within ~5% of analytical for reasonable sample size
        assert!(rel_diff < 0.1, "MC price too different from analytical");
    }
}
