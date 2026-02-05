//! Option quote data
//!
//! Market data for options including prices, implied volatility, and Greeks.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use super::option::{OptionContract, OptionType};
use super::greeks::Greeks;

/// Option market quote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionQuote {
    /// The option contract
    pub contract: OptionContract,
    /// Bid price
    pub bid: Option<f64>,
    /// Ask price
    pub ask: Option<f64>,
    /// Last traded price
    pub last: Option<f64>,
    /// Mid price (computed or provided)
    pub mid: Option<f64>,
    /// Trading volume
    pub volume: Option<u64>,
    /// Open interest
    pub open_interest: Option<u64>,
    /// Implied volatility (if provided by exchange)
    pub implied_vol: Option<f64>,
    /// Greeks (if provided or computed)
    pub greeks: Option<Greeks>,
    /// Quote timestamp
    pub timestamp: DateTime<Utc>,
    /// Underlying spot at quote time
    pub underlying_price: Option<f64>,
}

impl OptionQuote {
    /// Create a new quote
    pub fn new(contract: OptionContract) -> Self {
        Self {
            contract,
            bid: None,
            ask: None,
            last: None,
            mid: None,
            volume: None,
            open_interest: None,
            implied_vol: None,
            greeks: None,
            timestamp: Utc::now(),
            underlying_price: None,
        }
    }

    /// Compute mid price from bid/ask
    pub fn compute_mid(&mut self) {
        if let (Some(bid), Some(ask)) = (self.bid, self.ask) {
            self.mid = Some((bid + ask) / 2.0);
        }
    }

    /// Get the best available price (mid > last > bid)
    pub fn best_price(&self) -> Option<f64> {
        self.mid.or(self.last).or(self.bid)
    }

    /// Bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.bid, self.ask) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    /// Relative spread (spread / mid)
    pub fn relative_spread(&self) -> Option<f64> {
        match (self.spread(), self.mid) {
            (Some(s), Some(m)) if m > 0.0 => Some(s / m),
            _ => None,
        }
    }

    /// Is the quote valid for calibration?
    pub fn is_valid_for_calibration(&self) -> bool {
        // Need bid, ask with reasonable spread
        let has_prices = self.bid.is_some() && self.ask.is_some();
        let reasonable_spread = self.relative_spread().map(|s| s < 0.5).unwrap_or(false);
        let has_liquidity = self.open_interest.map(|oi| oi > 10).unwrap_or(true);
        let positive_price = self.best_price().map(|p| p > 0.01).unwrap_or(false);

        has_prices && reasonable_spread && has_liquidity && positive_price
    }
}

/// Chain of quotes for a single expiry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteChain {
    /// Underlying symbol
    pub underlying: String,
    /// Underlying spot price
    pub spot: f64,
    /// Expiry date
    pub expiry: chrono::NaiveDate,
    /// Time to expiry in years
    pub time_to_expiry: f64,
    /// Call quotes by strike
    pub calls: Vec<OptionQuote>,
    /// Put quotes by strike
    pub puts: Vec<OptionQuote>,
    /// Risk-free rate used
    pub risk_free_rate: f64,
    /// Dividend yield used
    pub dividend_yield: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl QuoteChain {
    pub fn new(underlying: impl Into<String>, spot: f64, expiry: chrono::NaiveDate) -> Self {
        let today = Utc::now().date_naive();
        let tte = (expiry - today).num_days() as f64 / 365.25;

        Self {
            underlying: underlying.into(),
            spot,
            expiry,
            time_to_expiry: tte,
            calls: Vec::new(),
            puts: Vec::new(),
            risk_free_rate: 0.05, // Default 5%
            dividend_yield: 0.01, // Default 1%
            timestamp: Utc::now(),
        }
    }

    /// Add a call quote
    pub fn add_call(&mut self, quote: OptionQuote) {
        self.calls.push(quote);
        self.calls.sort_by(|a, b| {
            a.contract.strike.partial_cmp(&b.contract.strike).unwrap()
        });
    }

    /// Add a put quote
    pub fn add_put(&mut self, quote: OptionQuote) {
        self.puts.push(quote);
        self.puts.sort_by(|a, b| {
            a.contract.strike.partial_cmp(&b.contract.strike).unwrap()
        });
    }

    /// Get all strikes
    pub fn strikes(&self) -> Vec<f64> {
        let mut strikes: Vec<f64> = self.calls.iter()
            .map(|q| q.contract.strike)
            .chain(self.puts.iter().map(|q| q.contract.strike))
            .collect();
        strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        strikes.dedup();
        strikes
    }

    /// Get ATM strike
    pub fn atm_strike(&self) -> f64 {
        self.strikes()
            .into_iter()
            .min_by(|a, b| {
                let da = (self.spot - a).abs();
                let db = (self.spot - b).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap_or(self.spot)
    }

    /// Get call quote at strike
    pub fn call_at(&self, strike: f64) -> Option<&OptionQuote> {
        self.calls.iter().find(|q| (q.contract.strike - strike).abs() < 0.01)
    }

    /// Get put quote at strike
    pub fn put_at(&self, strike: f64) -> Option<&OptionQuote> {
        self.puts.iter().find(|q| (q.contract.strike - strike).abs() < 0.01)
    }

    /// Get quotes valid for calibration
    pub fn calibration_quotes(&self) -> Vec<&OptionQuote> {
        self.calls.iter()
            .chain(self.puts.iter())
            .filter(|q| q.is_valid_for_calibration())
            .collect()
    }

    /// Forward price
    pub fn forward(&self) -> f64 {
        self.spot * ((self.risk_free_rate - self.dividend_yield) * self.time_to_expiry).exp()
    }
}

/// Full quote surface (all expiries)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteSurface {
    /// Underlying symbol
    pub underlying: String,
    /// Spot price
    pub spot: f64,
    /// All quote chains
    pub chains: Vec<QuoteChain>,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Dividend yield
    pub dividend_yield: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl QuoteSurface {
    pub fn new(underlying: impl Into<String>, spot: f64) -> Self {
        Self {
            underlying: underlying.into(),
            spot,
            chains: Vec::new(),
            risk_free_rate: 0.05,
            dividend_yield: 0.01,
            timestamp: Utc::now(),
        }
    }

    /// Add a chain
    pub fn add_chain(&mut self, mut chain: QuoteChain) {
        chain.risk_free_rate = self.risk_free_rate;
        chain.dividend_yield = self.dividend_yield;
        self.chains.push(chain);
        self.chains.sort_by_key(|c| c.expiry);
    }

    /// All expiries
    pub fn expiries(&self) -> Vec<chrono::NaiveDate> {
        self.chains.iter().map(|c| c.expiry).collect()
    }

    /// Times to expiry
    pub fn times_to_expiry(&self) -> Vec<f64> {
        self.chains.iter().map(|c| c.time_to_expiry).collect()
    }

    /// Get chain for expiry
    pub fn chain_for_expiry(&self, expiry: chrono::NaiveDate) -> Option<&QuoteChain> {
        self.chains.iter().find(|c| c.expiry == expiry)
    }

    /// Total number of quotes
    pub fn total_quotes(&self) -> usize {
        self.chains.iter().map(|c| c.calls.len() + c.puts.len()).sum()
    }

    /// Quotes valid for calibration
    pub fn calibration_quotes(&self) -> usize {
        self.chains.iter()
            .map(|c| c.calibration_quotes().len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn test_quote_validity() {
        let contract = OptionContract::european("QQQ", 500.0,
            NaiveDate::from_ymd_opt(2025, 6, 20).unwrap(), OptionType::Call);

        let mut quote = OptionQuote::new(contract);
        quote.bid = Some(10.0);
        quote.ask = Some(10.5);
        quote.open_interest = Some(1000);
        quote.compute_mid();

        assert!(quote.is_valid_for_calibration());
        assert!((quote.spread().unwrap() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_chain_forward() {
        // Use a future date 6 months from now
        let expiry = Utc::now().date_naive() + chrono::Duration::days(180);
        let chain = QuoteChain::new("QQQ", 500.0, expiry);

        // Forward should be slightly above spot with positive carry (rate > div)
        // With default rate=5%, div=1%, 6 months: forward = spot * exp((0.05-0.01)*0.5) ≈ 1.02 * spot
        assert!(chain.forward() > chain.spot * 1.01);
        assert!(chain.forward() < chain.spot * 1.05);
    }
}
