//! Option Greeks
//!
//! First and second order sensitivities for options.

use serde::{Deserialize, Serialize};

/// Option Greeks (sensitivities)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Greeks {
    /// Delta: dV/dS (sensitivity to spot)
    pub delta: f64,
    /// Gamma: d²V/dS² (sensitivity of delta to spot)
    pub gamma: f64,
    /// Theta: dV/dt (time decay, usually per day)
    pub theta: f64,
    /// Vega: dV/dσ (sensitivity to volatility)
    pub vega: f64,
    /// Rho: dV/dr (sensitivity to interest rate)
    pub rho: f64,
    /// Vanna: d²V/dSdσ (sensitivity of delta to vol)
    pub vanna: Option<f64>,
    /// Volga/Vomma: d²V/dσ² (sensitivity of vega to vol)
    pub volga: Option<f64>,
    /// Charm: d²V/dSdt (delta decay)
    pub charm: Option<f64>,
}

impl Greeks {
    pub fn new(delta: f64, gamma: f64, theta: f64, vega: f64, rho: f64) -> Self {
        Self {
            delta,
            gamma,
            theta,
            vega,
            rho,
            vanna: None,
            volga: None,
            charm: None,
        }
    }

    /// Scale Greeks by a factor (e.g., for notional)
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            delta: self.delta * factor,
            gamma: self.gamma * factor,
            theta: self.theta * factor,
            vega: self.vega * factor,
            rho: self.rho * factor,
            vanna: self.vanna.map(|v| v * factor),
            volga: self.volga.map(|v| v * factor),
            charm: self.charm.map(|v| v * factor),
        }
    }

    /// Add two Greeks (for portfolio)
    pub fn add(&self, other: &Greeks) -> Self {
        Self {
            delta: self.delta + other.delta,
            gamma: self.gamma + other.gamma,
            theta: self.theta + other.theta,
            vega: self.vega + other.vega,
            rho: self.rho + other.rho,
            vanna: match (self.vanna, other.vanna) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) | (None, Some(a)) => Some(a),
                _ => None,
            },
            volga: match (self.volga, other.volga) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) | (None, Some(a)) => Some(a),
                _ => None,
            },
            charm: match (self.charm, other.charm) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) | (None, Some(a)) => Some(a),
                _ => None,
            },
        }
    }
}

/// Greeks for a portfolio position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioGreeks {
    /// Net delta (dollar delta)
    pub delta_dollars: f64,
    /// Net gamma (dollar gamma)
    pub gamma_dollars: f64,
    /// Net theta (per day)
    pub theta_dollars: f64,
    /// Net vega (per 1% vol move)
    pub vega_dollars: f64,
    /// Net rho (per 1% rate move)
    pub rho_dollars: f64,
    /// Number of positions
    pub num_positions: usize,
}

impl PortfolioGreeks {
    pub fn new() -> Self {
        Self {
            delta_dollars: 0.0,
            gamma_dollars: 0.0,
            theta_dollars: 0.0,
            vega_dollars: 0.0,
            rho_dollars: 0.0,
            num_positions: 0,
        }
    }

    /// Add a position's Greeks
    pub fn add_position(&mut self, greeks: &Greeks, quantity: f64, multiplier: f64, spot: f64) {
        self.delta_dollars += greeks.delta * quantity * multiplier * spot;
        self.gamma_dollars += greeks.gamma * quantity * multiplier * spot * spot / 100.0;
        self.theta_dollars += greeks.theta * quantity * multiplier;
        self.vega_dollars += greeks.vega * quantity * multiplier;
        self.rho_dollars += greeks.rho * quantity * multiplier;
        self.num_positions += 1;
    }
}

impl Default for PortfolioGreeks {
    fn default() -> Self {
        Self::new()
    }
}
