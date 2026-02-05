//! Local Volatility Model (Dupire)
//!
//! The local volatility model assumes volatility is a deterministic function
//! of spot price and time: σ(S, t).
//!
//! Dupire's formula derives the local volatility from the implied volatility surface:
//!
//! σ_local²(K, T) = (∂w/∂T) / (1 - (y/w)(∂w/∂y) + (1/4)(-1/4 - 1/w + y²/w²)(∂w/∂y)² + (1/2)(∂²w/∂y²))
//!
//! where:
//! - w = σ_impl² * T (total variance)
//! - y = ln(K/F) (log-moneyness)
//!
//! This model:
//! - Perfectly fits the vanilla smile (by construction)
//! - But produces unrealistic forward smile dynamics
//! - Is used as one component in SLV models

use ndarray::{Array1, Array2};
use crate::core::{VolSurface, VolGridType, SLVError, SLVResult};

/// Local volatility surface
#[derive(Debug, Clone)]
pub struct LocalVolSurface {
    /// Underlying symbol
    pub underlying: String,
    /// Reference spot
    pub spot: f64,
    /// Log-moneyness grid (y = ln(K/F))
    pub log_moneyness: Vec<f64>,
    /// Time grid
    pub times: Vec<f64>,
    /// Local variance grid (σ²_local)
    pub local_var: Array2<f64>,
    /// Risk-free rate
    pub rate: f64,
    /// Dividend yield
    pub div_yield: f64,
}

impl LocalVolSurface {
    /// Build local vol surface from implied vol surface using Dupire formula
    pub fn from_implied_vol(iv_surface: &VolSurface) -> SLVResult<Self> {
        let n_strikes = iv_surface.x_axis.len();
        let n_times = iv_surface.y_axis.len();

        if n_strikes < 3 || n_times < 2 {
            return Err(SLVError::calibration(
                "Need at least 3 strikes and 2 expiries for local vol"
            ));
        }

        // Convert to log-moneyness coordinates
        let mut log_moneyness = Vec::with_capacity(n_strikes);
        for (ti, &time) in iv_surface.y_axis.iter().enumerate() {
            let forward = iv_surface.forwards.get(ti).copied()
                .unwrap_or(iv_surface.spot * ((iv_surface.risk_free_rate - iv_surface.dividend_yield) * time).exp());

            if ti == 0 {
                for &strike in &iv_surface.x_axis {
                    log_moneyness.push((strike / forward).ln());
                }
            }
        }

        // Compute total variance surface: w(y, T) = σ²(y, T) * T
        let mut total_var = Array2::zeros((n_strikes, n_times));
        for si in 0..n_strikes {
            for ti in 0..n_times {
                let vol = iv_surface.vols[[si, ti]];
                let time = iv_surface.y_axis[ti];
                total_var[[si, ti]] = vol * vol * time;
            }
        }

        // Compute local variance using Dupire formula
        let mut local_var = Array2::zeros((n_strikes, n_times));

        for si in 1..n_strikes-1 {
            for ti in 0..n_times-1 {
                let y = log_moneyness[si];
                let w = total_var[[si, ti]];
                let time = iv_surface.y_axis[ti];

                if w < 1e-10 || time < 1e-6 {
                    local_var[[si, ti]] = iv_surface.vols[[si, ti]].powi(2);
                    continue;
                }

                // Numerical derivatives
                let dy = if si > 0 && si < n_strikes - 1 {
                    (log_moneyness[si + 1] - log_moneyness[si - 1]) / 2.0
                } else {
                    0.01
                };

                let dt = if ti < n_times - 1 {
                    iv_surface.y_axis[ti + 1] - iv_surface.y_axis[ti]
                } else {
                    0.01
                };

                // ∂w/∂T
                let dw_dt = if ti < n_times - 1 {
                    (total_var[[si, ti + 1]] - total_var[[si, ti]]) / dt
                } else {
                    total_var[[si, ti]] / time // Fallback
                };

                // ∂w/∂y
                let dw_dy = (total_var[[si + 1, ti]] - total_var[[si - 1, ti]]) / (2.0 * dy);

                // ∂²w/∂y²
                let d2w_dy2 = (total_var[[si + 1, ti]] - 2.0 * total_var[[si, ti]] + total_var[[si - 1, ti]]) / (dy * dy);

                // Dupire formula
                let numerator = dw_dt;

                let term1 = 1.0 - (y / w) * dw_dy;
                let term2 = 0.25 * (-0.25 - 1.0/w + y*y/(w*w)) * dw_dy * dw_dy;
                let term3 = 0.5 * d2w_dy2;

                let denominator = term1 + term2 + term3;

                if denominator > 1e-10 {
                    let lv = numerator / denominator;
                    // Clamp to reasonable range
                    local_var[[si, ti]] = lv.clamp(0.0001, 4.0);
                } else {
                    // Fallback to implied vol
                    local_var[[si, ti]] = iv_surface.vols[[si, ti]].powi(2);
                }
            }
        }

        // Fill edges
        for ti in 0..n_times {
            local_var[[0, ti]] = local_var[[1, ti]];
            local_var[[n_strikes-1, ti]] = local_var[[n_strikes-2, ti]];
        }

        Ok(Self {
            underlying: iv_surface.underlying.clone(),
            spot: iv_surface.spot,
            log_moneyness,
            times: iv_surface.y_axis.clone(),
            local_var,
            rate: iv_surface.risk_free_rate,
            div_yield: iv_surface.dividend_yield,
        })
    }

    /// Interpolate local volatility at (spot, time)
    pub fn local_vol(&self, spot: f64, time: f64) -> f64 {
        // Convert spot to log-moneyness
        let forward = self.spot * ((self.rate - self.div_yield) * time).exp();
        let y = (spot / forward).ln();

        // Find bracketing indices
        let (yi_lo, yi_hi, y_frac) = self.find_bracket(&self.log_moneyness, y);
        let (ti_lo, ti_hi, t_frac) = self.find_bracket(&self.times, time);

        // Bilinear interpolation of variance
        let v00 = self.local_var[[yi_lo, ti_lo]];
        let v10 = self.local_var[[yi_hi, ti_lo]];
        let v01 = self.local_var[[yi_lo, ti_hi]];
        let v11 = self.local_var[[yi_hi, ti_hi]];

        let v0 = v00 * (1.0 - y_frac) + v10 * y_frac;
        let v1 = v01 * (1.0 - y_frac) + v11 * y_frac;
        let var = v0 * (1.0 - t_frac) + v1 * t_frac;

        var.sqrt().max(0.01)
    }

    /// Find bracketing indices for interpolation
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

    /// Get local vol smile at given time
    pub fn smile_at_time(&self, time: f64) -> Vec<(f64, f64)> {
        let (ti_lo, ti_hi, t_frac) = self.find_bracket(&self.times, time);

        self.log_moneyness.iter().enumerate().map(|(yi, &y)| {
            let v0 = self.local_var[[yi, ti_lo]];
            let v1 = self.local_var[[yi, ti_hi]];
            let var = v0 * (1.0 - t_frac) + v1 * t_frac;
            (y, var.sqrt())
        }).collect()
    }
}

/// Simple local vol model for Monte Carlo simulation
pub struct LocalVolModel {
    /// Local vol surface
    surface: LocalVolSurface,
}

impl LocalVolModel {
    pub fn new(surface: LocalVolSurface) -> Self {
        Self { surface }
    }

    /// Simulate spot path using Euler scheme
    /// Returns vector of (time, spot) pairs
    pub fn simulate_path(
        &self,
        spot: f64,
        time_horizon: f64,
        n_steps: usize,
        rng: &mut impl rand::Rng,
    ) -> Vec<(f64, f64)> {
        use rand_distr::{Distribution, StandardNormal};

        let dt = time_horizon / n_steps as f64;
        let sqrt_dt = dt.sqrt();

        let mut path = Vec::with_capacity(n_steps + 1);
        let mut s = spot;
        let mut t = 0.0;

        path.push((t, s));

        for _ in 0..n_steps {
            let local_vol = self.surface.local_vol(s, t);
            let dw: f64 = Distribution::<f64>::sample(&StandardNormal, rng) * sqrt_dt;

            // Euler-Maruyama: dS = (r - q) * S * dt + σ_local * S * dW
            let drift = (self.surface.rate - self.surface.div_yield) * s * dt;
            let diffusion = local_vol * s * dw;

            s = (s + drift + diffusion).max(0.01);
            t += dt;

            path.push((t, s));
        }

        path
    }

    /// Monte Carlo price for European option
    pub fn price_european(
        &self,
        strike: f64,
        time: f64,
        is_call: bool,
        n_paths: usize,
        n_steps: usize,
    ) -> f64 {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let df = (-self.surface.rate * time).exp();

        let payoffs: f64 = (0..n_paths)
            .map(|_| {
                let path = self.simulate_path(self.surface.spot, time, n_steps, &mut rng);
                let final_spot = path.last().unwrap().1;

                if is_call {
                    (final_spot - strike).max(0.0)
                } else {
                    (strike - final_spot).max(0.0)
                }
            })
            .sum();

        df * payoffs / n_paths as f64
    }
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

        // Create a simple smile: ATM = 20%, wings higher
        let mut vols: Array2<f64> = Array2::zeros((5, 3));
        for (ti, _) in times.iter().enumerate() {
            for (si, &strike) in strikes.iter().enumerate() {
                let moneyness = (strike / 100.0_f64).ln();
                let atm_vol = 0.20;
                let skew = -0.1 * moneyness;
                let smile = 0.5 * moneyness * moneyness;
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
    fn test_local_vol_construction() {
        let iv_surface = create_test_iv_surface();
        let lv_surface = LocalVolSurface::from_implied_vol(&iv_surface).unwrap();

        // Local vol should be in reasonable range
        let lv_atm = lv_surface.local_vol(100.0, 0.5);
        assert!(lv_atm > 0.1 && lv_atm < 0.5);
    }

    #[test]
    fn test_local_vol_model() {
        let iv_surface = create_test_iv_surface();
        let lv_surface = LocalVolSurface::from_implied_vol(&iv_surface).unwrap();
        let model = LocalVolModel::new(lv_surface);

        // Price ATM call
        let price = model.price_european(100.0, 0.5, true, 10000, 100);

        // Should be roughly consistent with BS price at ATM vol
        // BS ATM call ~ 0.4 * S * σ * sqrt(T) ≈ 0.4 * 100 * 0.2 * 0.707 ≈ 5.66
        assert!(price > 4.0 && price < 8.0);
    }
}
