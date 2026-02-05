//! Yahoo Finance data fetcher
//!
//! Fetches free options data for QQQ (as NDX proxy) and other ETFs.
//! Uses Yahoo Finance's unofficial API.
//!
//! Note: This is for educational/research purposes. Yahoo Finance
//! data is delayed ~15 minutes and intended for personal use.

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::core::{
    OptionContract, OptionQuote, OptionType, ExerciseStyle,
    QuoteChain, QuoteSurface, Greeks, SLVError, SLVResult,
};

/// Yahoo Finance API client
pub struct YahooClient {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl YahooClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::builder()
                .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://query1.finance.yahoo.com/v7/finance".to_string(),
        }
    }

    /// Get current quote for a symbol
    pub fn get_quote(&self, symbol: &str) -> SLVResult<SpotQuote> {
        let url = format!(
            "{}/quote?symbols={}",
            self.base_url, symbol
        );

        let response: YahooQuoteResponse = self.client
            .get(&url)
            .send()
            .map_err(|e| SLVError::Network(e.to_string()))?
            .json()
            .map_err(|e| SLVError::Data(format!("Failed to parse quote: {}", e)))?;

        let result = response.quote_response.result
            .into_iter()
            .next()
            .ok_or_else(|| SLVError::Data("No quote data returned".into()))?;

        Ok(SpotQuote {
            symbol: symbol.to_string(),
            price: result.regular_market_price,
            bid: result.bid,
            ask: result.ask,
            timestamp: Utc::now(),
        })
    }

    /// Get available option expiration dates
    pub fn get_expirations(&self, symbol: &str) -> SLVResult<Vec<NaiveDate>> {
        let url = format!(
            "{}/options/{}",
            self.base_url, symbol
        );

        let response: YahooOptionsResponse = self.client
            .get(&url)
            .send()
            .map_err(|e| SLVError::Network(e.to_string()))?
            .json()
            .map_err(|e| SLVError::Data(format!("Failed to parse options: {}", e)))?;

        let chain = response.option_chain.result
            .into_iter()
            .next()
            .ok_or_else(|| SLVError::Data("No options data returned".into()))?;

        let expiries: Vec<NaiveDate> = chain.expiration_dates
            .iter()
            .filter_map(|&ts| {
                DateTime::from_timestamp(ts, 0)
                    .map(|dt| dt.date_naive())
            })
            .collect();

        Ok(expiries)
    }

    /// Get option chain for a specific expiration
    pub fn get_option_chain(&self, symbol: &str, expiry: NaiveDate) -> SLVResult<QuoteChain> {
        // Convert expiry to Unix timestamp
        let expiry_ts = expiry
            .and_hms_opt(16, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();

        let url = format!(
            "{}/options/{}?date={}",
            self.base_url, symbol, expiry_ts
        );

        let response: YahooOptionsResponse = self.client
            .get(&url)
            .send()
            .map_err(|e| SLVError::Network(e.to_string()))?
            .json()
            .map_err(|e| SLVError::Data(format!("Failed to parse options: {}", e)))?;

        let chain_data = response.option_chain.result
            .into_iter()
            .next()
            .ok_or_else(|| SLVError::Data("No options data returned".into()))?;

        let spot = chain_data.quote.regular_market_price;
        let mut chain = QuoteChain::new(symbol, spot, expiry);

        // Process calls
        if let Some(options) = chain_data.options.first() {
            for call in &options.calls {
                if let Some(quote) = self.convert_option_quote(call, symbol, expiry, OptionType::Call) {
                    chain.add_call(quote);
                }
            }

            for put in &options.puts {
                if let Some(quote) = self.convert_option_quote(put, symbol, expiry, OptionType::Put) {
                    chain.add_put(quote);
                }
            }
        }

        Ok(chain)
    }

    /// Get full option surface (all expirations)
    pub fn get_option_surface(&self, symbol: &str) -> SLVResult<QuoteSurface> {
        let spot_quote = self.get_quote(symbol)?;
        let expiries = self.get_expirations(symbol)?;

        let mut surface = QuoteSurface::new(symbol, spot_quote.price);

        for expiry in expiries {
            match self.get_option_chain(symbol, expiry) {
                Ok(chain) => surface.add_chain(chain),
                Err(e) => {
                    tracing::warn!("Failed to get chain for {}: {}", expiry, e);
                }
            }
        }

        Ok(surface)
    }

    /// Convert Yahoo option data to our quote format
    fn convert_option_quote(
        &self,
        data: &YahooOptionData,
        underlying: &str,
        expiry: NaiveDate,
        option_type: OptionType,
    ) -> Option<OptionQuote> {
        let strike = data.strike?;

        let contract = OptionContract {
            underlying: underlying.to_string(),
            strike,
            expiry,
            option_type,
            exercise: ExerciseStyle::American, // US equity options are American
            multiplier: 100.0,
            symbol: data.contract_symbol.clone(),
        };

        let mut quote = OptionQuote::new(contract);
        quote.bid = data.bid;
        quote.ask = data.ask;
        quote.last = data.last_price;
        quote.volume = data.volume.map(|v| v as u64);
        quote.open_interest = data.open_interest.map(|oi| oi as u64);
        quote.implied_vol = data.implied_volatility;

        // Compute mid
        quote.compute_mid();

        // Convert Greeks if available
        if data.delta.is_some() || data.gamma.is_some() {
            quote.greeks = Some(Greeks {
                delta: data.delta.unwrap_or(0.0),
                gamma: data.gamma.unwrap_or(0.0),
                theta: data.theta.unwrap_or(0.0),
                vega: data.vega.unwrap_or(0.0),
                rho: data.rho.unwrap_or(0.0),
                vanna: None,
                volga: None,
                charm: None,
            });
        }

        Some(quote)
    }
}

impl Default for YahooClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Spot price quote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotQuote {
    pub symbol: String,
    pub price: f64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

// Yahoo Finance API response structures

#[derive(Debug, Deserialize)]
struct YahooQuoteResponse {
    #[serde(rename = "quoteResponse")]
    quote_response: YahooQuoteResult,
}

#[derive(Debug, Deserialize)]
struct YahooQuoteResult {
    result: Vec<YahooQuoteData>,
}

#[derive(Debug, Deserialize)]
struct YahooQuoteData {
    #[serde(rename = "regularMarketPrice")]
    regular_market_price: f64,
    bid: Option<f64>,
    ask: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct YahooOptionsResponse {
    #[serde(rename = "optionChain")]
    option_chain: YahooOptionChain,
}

#[derive(Debug, Deserialize)]
struct YahooOptionChain {
    result: Vec<YahooOptionChainData>,
}

#[derive(Debug, Deserialize)]
struct YahooOptionChainData {
    #[serde(rename = "expirationDates")]
    expiration_dates: Vec<i64>,
    quote: YahooQuoteData,
    options: Vec<YahooOptions>,
}

#[derive(Debug, Deserialize)]
struct YahooOptions {
    calls: Vec<YahooOptionData>,
    puts: Vec<YahooOptionData>,
}

#[derive(Debug, Deserialize)]
struct YahooOptionData {
    #[serde(rename = "contractSymbol")]
    contract_symbol: Option<String>,
    strike: Option<f64>,
    bid: Option<f64>,
    ask: Option<f64>,
    #[serde(rename = "lastPrice")]
    last_price: Option<f64>,
    volume: Option<i64>,
    #[serde(rename = "openInterest")]
    open_interest: Option<i64>,
    #[serde(rename = "impliedVolatility")]
    implied_volatility: Option<f64>,
    // Greeks (may not always be present)
    delta: Option<f64>,
    gamma: Option<f64>,
    theta: Option<f64>,
    vega: Option<f64>,
    rho: Option<f64>,
}

/// Convenience function to fetch QQQ options
pub fn fetch_qqq_options() -> SLVResult<QuoteSurface> {
    let client = YahooClient::new();
    client.get_option_surface("QQQ")
}

/// Convenience function to fetch SPY options
pub fn fetch_spy_options() -> SLVResult<QuoteSurface> {
    let client = YahooClient::new();
    client.get_option_surface("SPY")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network
    fn test_get_quote() {
        let client = YahooClient::new();
        let quote = client.get_quote("QQQ").unwrap();

        assert!(quote.price > 0.0);
        println!("QQQ price: {}", quote.price);
    }

    #[test]
    #[ignore] // Requires network
    fn test_get_expirations() {
        let client = YahooClient::new();
        let expiries = client.get_expirations("QQQ").unwrap();

        assert!(!expiries.is_empty());
        println!("QQQ expiries: {:?}", expiries);
    }

    #[test]
    #[ignore] // Requires network
    fn test_get_option_chain() {
        let client = YahooClient::new();
        let expiries = client.get_expirations("QQQ").unwrap();

        if let Some(&expiry) = expiries.first() {
            let chain = client.get_option_chain("QQQ", expiry).unwrap();

            println!("Chain for {}: {} calls, {} puts",
                expiry, chain.calls.len(), chain.puts.len());

            assert!(!chain.calls.is_empty());
            assert!(!chain.puts.is_empty());
        }
    }
}
