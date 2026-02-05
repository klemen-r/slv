//! Volatility Surface
//!
//! Implied volatility surface representation and interpolation.
//! Supports multiple parameterizations:
//! - Strike/Expiry grid
//! - Delta/Expiry grid
//! - Log-moneyness/Time grid

use chrono::NaiveDate;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Volatility surface representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolSurface {
    /// Underlying symbol
    pub underlying: String,
    /// Reference spot price
    pub spot: f64,
    /// Reference date
    pub reference_date: NaiveDate,
    /// Grid type
    pub grid_type: VolGridType,
    /// X-axis values (strikes, deltas, or log-moneyness)
    pub x_axis: Vec<f64>,
    /// Y-axis values (expiries as time in years)
    pub y_axis: Vec<f64>,
    /// Expiry dates (corresponding to y_axis)
    pub expiry_dates: Vec<NaiveDate>,
    /// Volatility grid [x, y] -> vol
    pub vols: Array2<f64>,
    /// Forward prices for each expiry
    pub forwards: Vec<f64>,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Dividend yield
    pub dividend_yield: f64,
}

/// Grid type for volatility surface
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolGridType {
    /// Absolute strike
    Strike,
    /// Delta (0 to 1 for calls, -1 to 0 for puts)
    Delta,
    /// Log-moneyness: ln(K/F)
    LogMoneyness,
    /// Standardized moneyness: ln(K/F) / sqrt(T)
    StandardizedMoneyness,
}

impl VolSurface {
    /// Create a new volatility surface
    pub fn new(
        underlying: impl Into<String>,
        spot: f64,
        reference_date: NaiveDate,
        grid_type: VolGridType,
    ) -> Self {
        Self {
            underlying: underlying.into(),
            spot,
            reference_date,
            grid_type,
            x_axis: Vec::new(),
            y_axis: Vec::new(),
            expiry_dates: Vec::new(),
            vols: Array2::zeros((0, 0)),
            forwards: Vec::new(),
            risk_free_rate: 0.05,
            dividend_yield: 0.01,
        }
    }

    /// Build surface from a grid of (strike, time, vol) points
    pub fn from_grid(
        underlying: impl Into<String>,
        spot: f64,
        reference_date: NaiveDate,
        strikes: Vec<f64>,
        times: Vec<f64>,
        expiries: Vec<NaiveDate>,
        vols: Array2<f64>,
        rate: f64,
        div_yield: f64,
    ) -> Self {
        let forwards: Vec<f64> = times
            .iter()
            .map(|&t| spot * ((rate - div_yield) * t).exp())
            .collect();

        Self {
            underlying: underlying.into(),
            spot,
            reference_date,
            grid_type: VolGridType::Strike,
            x_axis: strikes,
            y_axis: times,
            expiry_dates: expiries,
            vols,
            forwards,
            risk_free_rate: rate,
            dividend_yield: div_yield,
        }
    }

    /// Interpolate volatility at (strike, time)
    pub fn interpolate(&self, strike: f64, time: f64) -> Option<f64> {
        if self.x_axis.is_empty() || self.y_axis.is_empty() {
            return None;
        }

        // Find bracketing indices for x (strike)
        let (xi_lo, xi_hi, x_frac) = self.find_bracket(&self.x_axis, strike)?;

        // Find bracketing indices for y (time)
        let (yi_lo, yi_hi, y_frac) = self.find_bracket(&self.y_axis, time)?;

        // Bilinear interpolation
        let v00 = self.vols[[xi_lo, yi_lo]];
        let v10 = self.vols[[xi_hi, yi_lo]];
        let v01 = self.vols[[xi_lo, yi_hi]];
        let v11 = self.vols[[xi_hi, yi_hi]];

        let v0 = v00 * (1.0 - x_frac) + v10 * x_frac;
        let v1 = v01 * (1.0 - x_frac) + v11 * x_frac;

        Some(v0 * (1.0 - y_frac) + v1 * y_frac)
    }

    /// Find bracketing indices and interpolation fraction
    fn find_bracket(&self, axis: &[f64], value: f64) -> Option<(usize, usize, f64)> {
        if axis.is_empty() {
            return None;
        }

        // Clamp to bounds
        if value <= axis[0] {
            return Some((0, 0, 0.0));
        }
        if value >= axis[axis.len() - 1] {
            let last = axis.len() - 1;
            return Some((last, last, 0.0));
        }

        // Find bracket
        for i in 0..axis.len() - 1 {
            if value >= axis[i] && value <= axis[i + 1] {
                let frac = (value - axis[i]) / (axis[i + 1] - axis[i]);
                return Some((i, i + 1, frac));
            }
        }

        None
    }

    /// Get ATM volatility for a given time
    pub fn atm_vol(&self, time: f64) -> Option<f64> {
        // ATM strike is approximately the forward
        let forward = self.spot * ((self.risk_free_rate - self.dividend_yield) * time).exp();
        self.interpolate(forward, time)
    }

    /// Get volatility smile for a given time (all strikes)
    pub fn smile_at_time(&self, time: f64) -> Option<Vec<(f64, f64)>> {
        let (_, yi_hi, y_frac) = self.find_bracket(&self.y_axis, time)?;
        let yi_lo = if yi_hi > 0 { yi_hi - 1 } else { 0 };

        let smile: Vec<(f64, f64)> = self.x_axis
            .iter()
            .enumerate()
            .map(|(xi, &strike)| {
                let v_lo = self.vols[[xi, yi_lo]];
                let v_hi = self.vols[[xi, yi_hi]];
                let vol = v_lo * (1.0 - y_frac) + v_hi * y_frac;
                (strike, vol)
            })
            .collect();

        Some(smile)
    }

    /// Get term structure at ATM
    pub fn atm_term_structure(&self) -> Vec<(f64, f64)> {
        self.y_axis
            .iter()
            .filter_map(|&t| self.atm_vol(t).map(|v| (t, v)))
            .collect()
    }

    /// Convert to log-moneyness coordinates
    pub fn to_log_moneyness(&self) -> Self {
        if self.grid_type == VolGridType::LogMoneyness {
            return self.clone();
        }

        let mut new_x: Vec<f64> = Vec::new();
        let new_vols: Array2<f64> = Array2::zeros((self.x_axis.len(), self.y_axis.len()));

        // For each time, compute log-moneyness
        for (yi, &t) in self.y_axis.iter().enumerate() {
            let forward = self.forwards.get(yi).copied()
                .unwrap_or(self.spot * ((self.risk_free_rate - self.dividend_yield) * t).exp());

            for (xi, &strike) in self.x_axis.iter().enumerate() {
                let log_m = (strike / forward).ln();
                if yi == 0 {
                    new_x.push(log_m);
                }
            }
        }

        Self {
            underlying: self.underlying.clone(),
            spot: self.spot,
            reference_date: self.reference_date,
            grid_type: VolGridType::LogMoneyness,
            x_axis: new_x,
            y_axis: self.y_axis.clone(),
            expiry_dates: self.expiry_dates.clone(),
            vols: self.vols.clone(), // Same vol values, different x-axis interpretation
            forwards: self.forwards.clone(),
            risk_free_rate: self.risk_free_rate,
            dividend_yield: self.dividend_yield,
        }
    }

    /// Total variance at (strike, time): σ²T
    pub fn total_variance(&self, strike: f64, time: f64) -> Option<f64> {
        self.interpolate(strike, time).map(|v| v * v * time)
    }

    /// Local variance (Dupire): ∂w/∂T where w = σ²T
    pub fn local_variance(&self, strike: f64, time: f64, dt: f64) -> Option<f64> {
        let w1 = self.total_variance(strike, time)?;
        let w2 = self.total_variance(strike, time + dt)?;
        Some((w2 - w1) / dt)
    }
}

/// Smile parameterization using SABR-like representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmileParams {
    /// ATM volatility
    pub atm_vol: f64,
    /// Skew (slope at ATM)
    pub skew: f64,
    /// Curvature (smile convexity)
    pub curvature: f64,
    /// Put wing (extra convexity for low strikes)
    pub put_wing: f64,
    /// Call wing (extra convexity for high strikes)
    pub call_wing: f64,
}

impl SmileParams {
    /// Evaluate volatility at given log-moneyness
    pub fn vol_at(&self, log_moneyness: f64) -> f64 {
        // Simple SVI-like parameterization
        let k = log_moneyness;
        let base = self.atm_vol + self.skew * k;
        let smile = self.curvature * k * k;
        let wings = if k < 0.0 {
            self.put_wing * k * k * (-k).sqrt()
        } else {
            self.call_wing * k * k * k.sqrt()
        };

        (base + smile + wings).max(0.01)
    }

    /// Fit to market smile
    pub fn fit(log_moneyness: &[f64], vols: &[f64]) -> Option<Self> {
        if log_moneyness.len() < 3 {
            return None;
        }

        // Find ATM (closest to 0)
        let atm_idx = log_moneyness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())?
            .0;

        let atm_vol = vols[atm_idx];

        // Simple linear regression for skew
        let n = log_moneyness.len() as f64;
        let sum_x: f64 = log_moneyness.iter().sum();
        let sum_y: f64 = vols.iter().sum();
        let sum_xy: f64 = log_moneyness.iter().zip(vols.iter()).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = log_moneyness.iter().map(|x| x * x).sum();

        let skew = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        // Estimate curvature from residuals
        let residuals: Vec<f64> = log_moneyness
            .iter()
            .zip(vols.iter())
            .map(|(k, v)| v - atm_vol - skew * k)
            .collect();

        let curvature = residuals.iter().zip(log_moneyness.iter())
            .map(|(r, k)| if k.abs() > 0.01 { r / (k * k) } else { 0.0 })
            .sum::<f64>() / log_moneyness.len() as f64;

        Some(Self {
            atm_vol,
            skew,
            curvature: curvature.max(0.0),
            put_wing: 0.0,
            call_wing: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_interpolation() {
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let times = vec![0.25, 0.5, 1.0];
        let expiries = vec![
            NaiveDate::from_ymd_opt(2025, 4, 1).unwrap(),
            NaiveDate::from_ymd_opt(2025, 7, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
        ];

        // Flat vol surface at 20%
        let vols = Array2::from_elem((5, 3), 0.20);

        let surface = VolSurface::from_grid(
            "TEST", 100.0,
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            strikes, times, expiries, vols,
            0.05, 0.01,
        );

        // Interpolation should return ~0.20
        let vol = surface.interpolate(102.0, 0.4).unwrap();
        assert!((vol - 0.20).abs() < 0.001);
    }

    #[test]
    fn test_smile_params() {
        let params = SmileParams {
            atm_vol: 0.20,
            skew: -0.05,
            curvature: 0.02,
            put_wing: 0.01,
            call_wing: 0.005,
        };

        // ATM
        let atm = params.vol_at(0.0);
        assert!((atm - 0.20).abs() < 0.001);

        // OTM put (negative log-moneyness) should have higher vol (skew)
        let otm_put = params.vol_at(-0.1);
        assert!(otm_put > atm);
    }
}
