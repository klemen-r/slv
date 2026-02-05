//! Option contract definitions
//!
//! Represents vanilla European/American options with all contract specifications.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

/// Option type (Call or Put)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    /// Payoff direction: +1 for call, -1 for put
    pub fn phi(&self) -> f64 {
        match self {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        }
    }

    /// Intrinsic value at given spot
    pub fn intrinsic(&self, spot: f64, strike: f64) -> f64 {
        match self {
            OptionType::Call => (spot - strike).max(0.0),
            OptionType::Put => (strike - spot).max(0.0),
        }
    }
}

/// Exercise style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExerciseStyle {
    European,
    American,
}

/// Option contract specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionContract {
    /// Underlying symbol (e.g., "QQQ", "NQ", "NDX")
    pub underlying: String,
    /// Strike price
    pub strike: f64,
    /// Expiration date
    pub expiry: NaiveDate,
    /// Option type (Call/Put)
    pub option_type: OptionType,
    /// Exercise style
    pub exercise: ExerciseStyle,
    /// Contract multiplier (e.g., 100 for equity options)
    pub multiplier: f64,
    /// Contract symbol (exchange-specific)
    pub symbol: Option<String>,
}

impl OptionContract {
    /// Create a new European option
    pub fn european(
        underlying: impl Into<String>,
        strike: f64,
        expiry: NaiveDate,
        option_type: OptionType,
    ) -> Self {
        Self {
            underlying: underlying.into(),
            strike,
            expiry,
            option_type,
            exercise: ExerciseStyle::European,
            multiplier: 100.0,
            symbol: None,
        }
    }

    /// Create a new American option
    pub fn american(
        underlying: impl Into<String>,
        strike: f64,
        expiry: NaiveDate,
        option_type: OptionType,
    ) -> Self {
        Self {
            underlying: underlying.into(),
            strike,
            expiry,
            option_type,
            exercise: ExerciseStyle::American,
            multiplier: 100.0,
            symbol: None,
        }
    }

    /// Time to expiry in years from given date
    pub fn time_to_expiry(&self, from: NaiveDate) -> f64 {
        let days = (self.expiry - from).num_days();
        days as f64 / 365.25
    }

    /// Time to expiry from now
    pub fn time_to_expiry_now(&self) -> f64 {
        let today = Utc::now().date_naive();
        self.time_to_expiry(today)
    }

    /// Log-moneyness: ln(K/S)
    pub fn log_moneyness(&self, spot: f64) -> f64 {
        (self.strike / spot).ln()
    }

    /// Forward log-moneyness: ln(K/F) where F = S * exp(r*T)
    pub fn forward_log_moneyness(&self, spot: f64, rate: f64, time: f64) -> f64 {
        let forward = spot * (rate * time).exp();
        (self.strike / forward).ln()
    }

    /// Is this option in the money?
    pub fn is_itm(&self, spot: f64) -> bool {
        match self.option_type {
            OptionType::Call => spot > self.strike,
            OptionType::Put => spot < self.strike,
        }
    }

    /// Is this option at the money (within tolerance)?
    pub fn is_atm(&self, spot: f64, tolerance: f64) -> bool {
        (self.strike - spot).abs() / spot < tolerance
    }

    /// Is this option out of the money?
    pub fn is_otm(&self, spot: f64) -> bool {
        !self.is_itm(spot) && !self.is_atm(spot, 0.01)
    }
}

/// An option chain for a single expiry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionChain {
    /// Underlying symbol
    pub underlying: String,
    /// Expiration date
    pub expiry: NaiveDate,
    /// All strikes available
    pub strikes: Vec<f64>,
    /// Call contracts
    pub calls: Vec<OptionContract>,
    /// Put contracts
    pub puts: Vec<OptionContract>,
    /// Timestamp when fetched
    pub timestamp: DateTime<Utc>,
}

impl OptionChain {
    pub fn new(underlying: impl Into<String>, expiry: NaiveDate) -> Self {
        Self {
            underlying: underlying.into(),
            expiry,
            strikes: Vec::new(),
            calls: Vec::new(),
            puts: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    /// Add a strike with both call and put
    pub fn add_strike(&mut self, strike: f64) {
        if !self.strikes.contains(&strike) {
            self.strikes.push(strike);
            self.strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());

            self.calls.push(OptionContract::american(
                self.underlying.clone(),
                strike,
                self.expiry,
                OptionType::Call,
            ));
            self.puts.push(OptionContract::american(
                self.underlying.clone(),
                strike,
                self.expiry,
                OptionType::Put,
            ));
        }
    }

    /// Get ATM strike (closest to spot)
    pub fn atm_strike(&self, spot: f64) -> Option<f64> {
        self.strikes
            .iter()
            .min_by(|a, b| {
                let da = (spot - *a).abs();
                let db = (spot - *b).abs();
                da.partial_cmp(&db).unwrap()
            })
            .copied()
    }

    /// Get call at strike
    pub fn call_at(&self, strike: f64) -> Option<&OptionContract> {
        self.calls.iter().find(|c| (c.strike - strike).abs() < 0.001)
    }

    /// Get put at strike
    pub fn put_at(&self, strike: f64) -> Option<&OptionContract> {
        self.puts.iter().find(|p| (p.strike - strike).abs() < 0.001)
    }
}

/// Full option surface (multiple expiries)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionSurface {
    /// Underlying symbol
    pub underlying: String,
    /// Spot price at time of snapshot
    pub spot: f64,
    /// All chains by expiry
    pub chains: Vec<OptionChain>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl OptionSurface {
    pub fn new(underlying: impl Into<String>, spot: f64) -> Self {
        Self {
            underlying: underlying.into(),
            spot,
            chains: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    /// Add a chain
    pub fn add_chain(&mut self, chain: OptionChain) {
        self.chains.push(chain);
        self.chains.sort_by_key(|c| c.expiry);
    }

    /// Get chain for expiry
    pub fn chain_for_expiry(&self, expiry: NaiveDate) -> Option<&OptionChain> {
        self.chains.iter().find(|c| c.expiry == expiry)
    }

    /// All expiries
    pub fn expiries(&self) -> Vec<NaiveDate> {
        self.chains.iter().map(|c| c.expiry).collect()
    }

    /// Get nearest expiry to target days
    pub fn nearest_expiry(&self, target_days: i64) -> Option<NaiveDate> {
        let today = Utc::now().date_naive();
        self.chains
            .iter()
            .min_by_key(|c| ((c.expiry - today).num_days() - target_days).abs())
            .map(|c| c.expiry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_type() {
        assert_eq!(OptionType::Call.phi(), 1.0);
        assert_eq!(OptionType::Put.phi(), -1.0);

        assert_eq!(OptionType::Call.intrinsic(110.0, 100.0), 10.0);
        assert_eq!(OptionType::Put.intrinsic(90.0, 100.0), 10.0);
        assert_eq!(OptionType::Call.intrinsic(90.0, 100.0), 0.0);
    }

    #[test]
    fn test_time_to_expiry() {
        let expiry = NaiveDate::from_ymd_opt(2025, 6, 20).unwrap();
        let today = NaiveDate::from_ymd_opt(2025, 1, 20).unwrap();

        let opt = OptionContract::european("QQQ", 500.0, expiry, OptionType::Call);
        let tte = opt.time_to_expiry(today);

        // ~5 months = ~0.41 years
        assert!(tte > 0.4 && tte < 0.42);
    }

    #[test]
    fn test_moneyness() {
        let expiry = NaiveDate::from_ymd_opt(2025, 6, 20).unwrap();
        let opt = OptionContract::european("QQQ", 500.0, expiry, OptionType::Call);

        // ATM
        assert!(opt.is_atm(500.0, 0.01));
        // ITM call
        assert!(opt.is_itm(510.0));
        // OTM call
        assert!(opt.is_otm(490.0));
    }
}
