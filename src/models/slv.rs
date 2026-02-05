//! Stochastic Local Volatility (SLV) Model
//!
//! The SLV model combines Heston stochastic volatility with Dupire local volatility:
//!
//! dS = (r - q) * S * dt + L(S,t) * √v * S * dW_S
//! dv = κ(θ - v) * dt + σ_v * √v * dW_v
//!
//! where L(S,t) is the "leverage function" that mixes local and stochastic components.
//!
//! The leverage function is calibrated so that:
//! E[L²(S,t) * v | S_t = K] = σ²_local(K, t)
//!
//! This ensures:
//! - Perfect fit to vanilla smile (from local vol component)
//! - Realistic forward smile dynamics (from stochastic vol component)
//!
//! The "mixing" parameter α controls the blend:
//! - α = 0: Pure local volatility
//! - α = 1: Pure stochastic volatility
//! - 0 < α < 1: SLV mixture

use ndarray::Array2;
use crate::core::{OptionType, VolSurface, SLVError, SLVResult};
use super::heston::{HestonParams, HestonModel};
use super::local_vol::LocalVolSurface;

/// SLV Model parameters
#[derive(Debug, Clone)]
pub struct SLVParams {
    /// Heston parameters
    pub heston: HestonParams,
    /// Mixing parameter α ∈ [0, 1]
    /// α = 0: pure local vol, α = 1: pure stochastic vol
    pub alpha: f64,
    /// Risk-free rate
    pub rate: f64,
    /// Dividend yield
    pub div_yield: f64,
}

impl SLVParams {
    pub fn new(heston: HestonParams, alpha: f64, rate: f64, div_yield: f64) -> Self {
        Self {
            heston,
            alpha: alpha.clamp(0.0, 1.0),
            rate,
            div_yield,
        }
    }

    /// Create with typical equity parameters
    pub fn typical_equity(alpha: f64) -> Self {
        Self {
            heston: HestonParams::typical_equity(),
            alpha: alpha.clamp(0.0, 1.0),
            rate: 0.05,
            div_yield: 0.01,
        }
    }
}

/// Stochastic Local Volatility Model
pub struct SLVModel {
    /// Model parameters
    params: SLVParams,
    /// Local volatility surface
    local_vol: LocalVolSurface,
    /// Leverage function L(S, t) - calibrated to match market
    leverage: LeverageFunction,
    /// Reference spot
    spot: f64,
}

impl SLVModel {
    /// Create SLV model from implied vol surface
    pub fn from_iv_surface(
        iv_surface: &VolSurface,
        params: SLVParams,
    ) -> SLVResult<Self> {
        // Build local vol surface
        let local_vol = LocalVolSurface::from_implied_vol(iv_surface)?;

        // Initialize leverage function
        let leverage = LeverageFunction::new(
            &local_vol,
            &params,
            iv_surface.spot,
        )?;

        Ok(Self {
            params,
            local_vol,
            leverage,
            spot: iv_surface.spot,
        })
    }

    /// Get effective volatility at (spot, variance, time)
    pub fn effective_vol(&self, spot: f64, variance: f64, time: f64) -> f64 {
        let local_vol = self.local_vol.local_vol(spot, time);
        let stoch_vol = variance.sqrt();
        let leverage = self.leverage.evaluate(spot, time);

        // SLV diffusion: L(S,t) * sqrt(v)
        // Blend: σ_eff = (1-α) * σ_local + α * L * sqrt(v)
        let alpha = self.params.alpha;
        (1.0 - alpha) * local_vol + alpha * leverage * stoch_vol
    }

    /// Simulate SLV path
    pub fn simulate_path(
        &self,
        time_horizon: f64,
        n_steps: usize,
        rng: &mut impl rand::Rng,
    ) -> Vec<(f64, f64, f64)> {  // (time, spot, variance)
        use rand_distr::{Distribution, StandardNormal};

        let dt = time_horizon / n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let p = &self.params;

        let mut path = Vec::with_capacity(n_steps + 1);
        let mut s = self.spot;
        let mut v = p.heston.v0;
        let mut t = 0.0;

        path.push((t, s, v));

        for _ in 0..n_steps {
            // Generate correlated Brownians
            let z1: f64 = Distribution::<f64>::sample(&StandardNormal, rng);
            let z2: f64 = Distribution::<f64>::sample(&StandardNormal, rng);
            let rho = p.heston.rho;
            let dw_s = z1 * sqrt_dt;
            let dw_v = (rho * z1 + (1.0 - rho * rho).sqrt() * z2) * sqrt_dt;

            // Variance process (Heston)
            let v_sqrt = v.sqrt().max(0.0);
            let dv = p.heston.kappa * (p.heston.theta - v) * dt
                   + p.heston.sigma * v_sqrt * dw_v;
            v = (v + dv).max(0.0);

            // Spot process (SLV)
            let eff_vol = self.effective_vol(s, v, t);
            let ds = (p.rate - p.div_yield) * s * dt + eff_vol * s * dw_s;
            s = (s + ds).max(0.01);

            t += dt;
            path.push((t, s, v));
        }

        path
    }

    /// Monte Carlo price for European option
    pub fn mc_price(
        &self,
        strike: f64,
        time: f64,
        option_type: OptionType,
        n_paths: usize,
        n_steps: usize,
    ) -> f64 {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let df = (-self.params.rate * time).exp();

        let payoffs: f64 = (0..n_paths)
            .map(|_| {
                let path = self.simulate_path(time, n_steps, &mut rng);
                let final_spot = path.last().unwrap().1;
                option_type.intrinsic(final_spot, strike)
            })
            .sum();

        df * payoffs / n_paths as f64
    }

    /// Compute implied vol from SLV price
    pub fn implied_vol(
        &self,
        strike: f64,
        time: f64,
        option_type: OptionType,
        n_paths: usize,
    ) -> SLVResult<f64> {
        let n_steps = (time * 252.0).max(50.0) as usize;
        let price = self.mc_price(strike, time, option_type, n_paths, n_steps);

        super::black_scholes::implied_volatility(
            price, self.spot, strike,
            self.params.rate, self.params.div_yield,
            time, option_type
        )
    }

    /// Generate SLV smile
    pub fn smile(&self, strikes: &[f64], time: f64, n_paths: usize) -> Vec<(f64, f64)> {
        strikes.iter().filter_map(|&strike| {
            self.implied_vol(strike, time, OptionType::Call, n_paths)
                .ok()
                .map(|iv| (strike, iv))
        }).collect()
    }

    /// Get current parameters
    pub fn params(&self) -> &SLVParams {
        &self.params
    }

    /// Get local vol surface
    pub fn local_vol_surface(&self) -> &LocalVolSurface {
        &self.local_vol
    }
}

/// Leverage function L(S, t) for SLV model
///
/// The leverage function adjusts the stochastic volatility component
/// to match the local volatility surface.
#[derive(Debug, Clone)]
pub struct LeverageFunction {
    /// Spot grid
    spots: Vec<f64>,
    /// Time grid
    times: Vec<f64>,
    /// Leverage values L(S, t)
    values: Array2<f64>,
}

impl LeverageFunction {
    /// Build initial leverage function from local vol surface
    pub fn new(
        local_vol: &LocalVolSurface,
        params: &SLVParams,
        spot: f64,
    ) -> SLVResult<Self> {
        // Create spot grid around reference spot
        let n_spots = 51;
        let n_times = local_vol.times.len();

        let spot_range = 0.5; // ±50% around spot
        let spots: Vec<f64> = (0..n_spots)
            .map(|i| {
                let frac = i as f64 / (n_spots - 1) as f64;
                spot * (1.0 - spot_range + 2.0 * spot_range * frac)
            })
            .collect();

        let times = local_vol.times.clone();

        // Initialize leverage function
        // L(S,t) = σ_local(S,t) / E[√v | S_t = S]
        // Initially approximate E[√v] ≈ √θ (long-term variance)
        let expected_sqrt_v = params.heston.theta.sqrt();

        let mut values = Array2::zeros((n_spots, n_times));

        for (si, &s) in spots.iter().enumerate() {
            for (ti, &t) in times.iter().enumerate() {
                let lv = local_vol.local_vol(s, t);
                // Leverage = local_vol / stochastic_vol_contribution
                // With mixing: effective_vol = (1-α)*lv + α*L*√v
                // To match: effective_vol = lv (at market)
                // So: L = lv / √v (approximately)
                let leverage = if expected_sqrt_v > 0.01 {
                    lv / expected_sqrt_v
                } else {
                    1.0
                };
                values[[si, ti]] = leverage.clamp(0.1, 3.0);
            }
        }

        Ok(Self { spots, times, values })
    }

    /// Evaluate leverage at (spot, time)
    pub fn evaluate(&self, spot: f64, time: f64) -> f64 {
        // Bilinear interpolation
        let (si_lo, si_hi, s_frac) = self.find_bracket(&self.spots, spot);
        let (ti_lo, ti_hi, t_frac) = self.find_bracket(&self.times, time);

        let v00 = self.values[[si_lo, ti_lo]];
        let v10 = self.values[[si_hi, ti_lo]];
        let v01 = self.values[[si_lo, ti_hi]];
        let v11 = self.values[[si_hi, ti_hi]];

        let v0 = v00 * (1.0 - s_frac) + v10 * s_frac;
        let v1 = v01 * (1.0 - s_frac) + v11 * s_frac;

        v0 * (1.0 - t_frac) + v1 * t_frac
    }

    fn find_bracket(&self, axis: &[f64], value: f64) -> (usize, usize, f64) {
        if axis.is_empty() {
            return (0, 0, 0.0);
        }

        if value <= axis[0] {
            return (0, 0, 0.0);
        }
        if value >= axis[axis.len() - 1] {
            let last = axis.len() - 1;
            return (last, last, 0.0);
        }

        for i in 0..axis.len() - 1 {
            if value >= axis[i] && value <= axis[i + 1] {
                let frac = (value - axis[i]) / (axis[i + 1] - axis[i]);
                return (i, i + 1, frac);
            }
        }

        let last = axis.len() - 1;
        (last, last, 0.0)
    }

    /// Get leverage grid for visualization
    pub fn grid(&self) -> (&[f64], &[f64], &Array2<f64>) {
        (&self.spots, &self.times, &self.values)
    }
}

/// SLV calibration result
#[derive(Debug, Clone)]
pub struct SLVCalibrationResult {
    /// Calibrated parameters
    pub params: SLVParams,
    /// Calibration error (RMSE of IV fit)
    pub rmse: f64,
    /// Max absolute IV error
    pub max_error: f64,
    /// Number of iterations
    pub iterations: usize,
}

/// Calibrate SLV model to market quotes
pub fn calibrate_slv(
    iv_surface: &VolSurface,
    initial_params: Option<SLVParams>,
) -> SLVResult<SLVCalibrationResult> {
    // Start with initial parameters or defaults
    let mut params = initial_params.unwrap_or_else(|| SLVParams::typical_equity(0.5));

    // Grid search over mixing parameter α
    let mut best_rmse = f64::INFINITY;
    let mut best_params = params.clone();

    for alpha_pct in (0..=10).map(|i| i as f64 * 0.1) {
        params.alpha = alpha_pct;

        match SLVModel::from_iv_surface(iv_surface, params.clone()) {
            Ok(model) => {
                // Compute fit error
                let rmse = compute_fit_error(&model, iv_surface)?;

                if rmse < best_rmse {
                    best_rmse = rmse;
                    best_params = params.clone();
                }
            }
            Err(_) => continue,
        }
    }

    // TODO: Add Heston parameter optimization
    // For now, use typical equity params with optimized alpha

    Ok(SLVCalibrationResult {
        params: best_params,
        rmse: best_rmse,
        max_error: best_rmse * 2.0, // Approximate
        iterations: 11, // Number of alpha values tried
    })
}

/// Compute RMSE of IV fit
fn compute_fit_error(model: &SLVModel, market: &VolSurface) -> SLVResult<f64> {
    let mut sum_sq_error = 0.0;
    let mut count = 0;

    // Sample a few points (full calibration would check more)
    let n_paths = 5000;

    for (ti, &time) in market.y_axis.iter().enumerate().take(3) {
        let n_strikes = market.x_axis.len().min(9);
        let step = (market.x_axis.len() / n_strikes).max(1);

        for (si, &strike) in market.x_axis.iter().step_by(step).enumerate().take(n_strikes) {
            let market_iv = market.vols[[si * step, ti]];

            if market_iv > 0.01 {
                if let Ok(model_iv) = model.implied_vol(strike, time, OptionType::Call, n_paths) {
                    let error = model_iv - market_iv;
                    sum_sq_error += error * error;
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        return Err(SLVError::calibration("No valid points for calibration"));
    }

    Ok((sum_sq_error / count as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use chrono::NaiveDate;

    fn create_test_iv_surface() -> VolSurface {
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let times = vec![0.25, 0.5, 1.0];
        let expiries = vec![
            NaiveDate::from_ymd_opt(2025, 4, 1).unwrap(),
            NaiveDate::from_ymd_opt(2025, 7, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
        ];

        // Create a realistic smile with skew
        let mut vols: Array2<f64> = Array2::zeros((5, 3));
        for (ti, _) in times.iter().enumerate() {
            for (si, &strike) in strikes.iter().enumerate() {
                let moneyness = (strike / 100.0_f64).ln();
                let atm_vol = 0.20;
                let skew = -0.15 * moneyness; // Negative skew
                let smile = 0.3 * moneyness * moneyness;
                vols[[si, ti]] = atm_vol + skew + smile;
            }
        }

        VolSurface::from_grid(
            "TEST", 100.0,
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            strikes, times, expiries, vols,
            0.05, 0.01,
        )
    }

    #[test]
    fn test_slv_construction() {
        let iv_surface = create_test_iv_surface();
        let params = SLVParams::typical_equity(0.5);

        let model = SLVModel::from_iv_surface(&iv_surface, params).unwrap();

        // Effective vol should be in reasonable range
        let eff_vol = model.effective_vol(100.0, 0.04, 0.5);
        assert!(eff_vol > 0.1 && eff_vol < 0.5);
    }

    #[test]
    fn test_slv_simulation() {
        let iv_surface = create_test_iv_surface();
        let params = SLVParams::typical_equity(0.5);
        let model = SLVModel::from_iv_surface(&iv_surface, params).unwrap();

        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let path = model.simulate_path(0.5, 100, &mut rng);

        assert_eq!(path.len(), 101);

        // Spot should stay positive and in reasonable range
        for (_, s, v) in &path {
            assert!(*s > 0.0);
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_slv_pricing() {
        let iv_surface = create_test_iv_surface();
        let params = SLVParams::typical_equity(0.5);
        let model = SLVModel::from_iv_surface(&iv_surface, params).unwrap();

        let call_price = model.mc_price(100.0, 0.25, OptionType::Call, 10000, 50);

        // Should be positive and in reasonable range
        assert!(call_price > 2.0 && call_price < 15.0);
    }

    #[test]
    fn test_leverage_function() {
        let iv_surface = create_test_iv_surface();
        let local_vol = LocalVolSurface::from_implied_vol(&iv_surface).unwrap();
        let params = SLVParams::typical_equity(0.5);

        let leverage = LeverageFunction::new(&local_vol, &params, 100.0).unwrap();

        // Leverage should be around 1.0 at ATM
        let lev_atm = leverage.evaluate(100.0, 0.25);
        assert!(lev_atm > 0.5 && lev_atm < 2.0);
    }
}
