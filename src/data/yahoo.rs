//! Yahoo Finance data fetcher
//!
//! Fetches free options data for QQQ (as NDX proxy) and other ETFs.
//! Uses Yahoo Finance's unofficial API.
//!
//! Note: This is for educational/research purposes. Yahoo Finance
//! data is delayed ~15 minutes and intended for personal use.

use chrono::{DateTime, NaiveDate, Utc};
use reqwest::Url;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

use crate::core::{
    ExerciseStyle, Greeks, OptionContract, OptionQuote, OptionType, QuoteChain, QuoteSurface,
    SLVError, SLVResult,
};

/// Yahoo Finance API client
pub struct YahooClient {
    client: reqwest::blocking::Client,
    base_url: String,
    crumb: Mutex<Option<String>>,
}

impl YahooClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::builder()
                .cookie_store(true)
                .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://query1.finance.yahoo.com/v7/finance".to_string(),
            crumb: Mutex::new(None),
        }
    }

    fn reset_crumb(&self) {
        if let Ok(mut guard) = self.crumb.lock() {
            *guard = None;
        }
    }

    fn ensure_crumb(&self) -> SLVResult<String> {
        if let Ok(guard) = self.crumb.lock() {
            if let Some(existing) = guard.clone() {
                return Ok(existing);
            }
        }

        self.client
            .get("https://fc.yahoo.com")
            .send()
            .map_err(|e| SLVError::Network(format!("Failed to get Yahoo cookie: {}", e)))?;

        let crumb_raw = self
            .client
            .get("https://query1.finance.yahoo.com/v1/test/getcrumb")
            .send()
            .map_err(|e| SLVError::Network(format!("Failed to get Yahoo crumb: {}", e)))?
            .text()
            .map_err(|e| {
                SLVError::Network(format!("Failed reading Yahoo crumb response: {}", e))
            })?;

        let crumb = crumb_raw.trim().to_string();
        if crumb.is_empty() || crumb.starts_with('{') || crumb.contains("Invalid") {
            return Err(SLVError::Data(format!(
                "Failed to obtain Yahoo crumb: {}",
                crumb_raw.trim()
            )));
        }

        let mut guard = self
            .crumb
            .lock()
            .map_err(|_| SLVError::Data("Yahoo crumb lock poisoned".into()))?;
        *guard = Some(crumb.clone());
        Ok(crumb)
    }

    fn get_json_with_crumb<T>(&self, url: &str, what: &str) -> SLVResult<T>
    where
        T: DeserializeOwned,
    {
        let mut retried = false;

        loop {
            let crumb = self.ensure_crumb()?;
            let mut full_url = Url::parse(url)
                .map_err(|e| SLVError::Data(format!("Invalid Yahoo URL {}: {}", url, e)))?;
            full_url.query_pairs_mut().append_pair("crumb", &crumb);

            let response = self
                .client
                .get(full_url.clone())
                .send()
                .map_err(|e| SLVError::Network(e.to_string()))?;

            if response.status() == reqwest::StatusCode::UNAUTHORIZED && !retried {
                retried = true;
                self.reset_crumb();
                continue;
            }

            let status = response.status();
            let body = response
                .text()
                .map_err(|e| SLVError::Network(e.to_string()))?;

            if !status.is_success() {
                let snippet: String = body.chars().take(240).collect();
                return Err(SLVError::Network(format!(
                    "Yahoo {} request failed ({}): {}",
                    what, status, snippet
                )));
            }

            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) {
                if let Some(finance_error) = value
                    .get("finance")
                    .and_then(|f| f.get("error"))
                    .and_then(|e| e.as_object())
                {
                    let code = finance_error
                        .get("code")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown");
                    let description = finance_error
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown Yahoo error");
                    return Err(SLVError::Data(format!(
                        "Yahoo {} error [{}]: {}",
                        what, code, description
                    )));
                }
            }

            return serde_json::from_str::<T>(&body).map_err(|e| {
                let snippet: String = body.chars().take(240).collect();
                SLVError::Data(format!(
                    "Failed to parse {}: {} | body: {}",
                    what, e, snippet
                ))
            });
        }
    }

    fn options_symbol_candidates(symbol: &str) -> Vec<String> {
        let normalized = symbol.trim().to_uppercase();
        if normalized.is_empty() {
            return vec![];
        }

        let mut candidates = vec![normalized.clone()];

        if normalized.starts_with('^') {
            let without_caret = normalized.trim_start_matches('^').to_string();
            if !without_caret.is_empty() && without_caret != normalized {
                candidates.push(without_caret);
            }
        } else {
            candidates.push(format!("^{}", normalized));
        }

        candidates
    }

    fn get_expirations_for_symbol(&self, yahoo_symbol: &str) -> SLVResult<Vec<NaiveDate>> {
        let url = format!("{}/options/{}", self.base_url, yahoo_symbol);
        let response: YahooOptionsResponse = self.get_json_with_crumb(&url, "options")?;

        let chain = response
            .option_chain
            .result
            .into_iter()
            .next()
            .ok_or_else(|| SLVError::Data("No options data returned".into()))?;

        let expiries: Vec<NaiveDate> = chain
            .expiration_dates
            .iter()
            .filter_map(|&ts| DateTime::from_timestamp(ts, 0).map(|dt| dt.date_naive()))
            .collect();

        Ok(expiries)
    }

    fn get_option_chain_for_symbol(
        &self,
        requested_symbol: &str,
        yahoo_symbol: &str,
        expiry: NaiveDate,
    ) -> SLVResult<QuoteChain> {
        let expiry_ts = expiry.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
        let url = format!(
            "{}/options/{}?date={}",
            self.base_url, yahoo_symbol, expiry_ts
        );
        let response: YahooOptionsResponse = self.get_json_with_crumb(&url, "options chain")?;

        let chain_data = response
            .option_chain
            .result
            .into_iter()
            .next()
            .ok_or_else(|| SLVError::Data("No options data returned".into()))?;

        let spot = chain_data.quote.regular_market_price;
        let mut chain = QuoteChain::new(requested_symbol, spot, expiry);

        if let Some(options) = chain_data.options.first() {
            for call in &options.calls {
                if let Some(quote) =
                    self.convert_option_quote(call, requested_symbol, expiry, OptionType::Call)
                {
                    chain.add_call(quote);
                }
            }

            for put in &options.puts {
                if let Some(quote) =
                    self.convert_option_quote(put, requested_symbol, expiry, OptionType::Put)
                {
                    chain.add_put(quote);
                }
            }
        }

        Ok(chain)
    }

    fn get_candles_for_symbol(
        &self,
        yahoo_symbol: &str,
        interval: &str,
        range: &str,
    ) -> SLVResult<Vec<CandleBar>> {
        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval={}&range={}&includePrePost=false",
            yahoo_symbol, interval, range
        );
        let response: YahooChartResponse = self.get_json_with_crumb(&url, "chart")?;

        let chart_error_message = response.chart.error.as_ref().map(|e| {
            format!(
                "{}: {}",
                e.code.as_deref().unwrap_or("Unknown"),
                e.description
            )
        });

        let chart_result = response
            .chart
            .result
            .and_then(|mut list| list.pop())
            .ok_or_else(|| {
                SLVError::Data(
                    chart_error_message.unwrap_or_else(|| "No chart result returned".to_string()),
                )
            })?;

        let timestamps = chart_result
            .timestamp
            .ok_or_else(|| SLVError::Data("Chart response missing timestamps".into()))?;

        let quote = chart_result
            .indicators
            .quote
            .into_iter()
            .next()
            .ok_or_else(|| SLVError::Data("Chart response missing quote series".into()))?;

        let opens = quote
            .open
            .ok_or_else(|| SLVError::Data("Chart response missing open series".into()))?;
        let highs = quote
            .high
            .ok_or_else(|| SLVError::Data("Chart response missing high series".into()))?;
        let lows = quote
            .low
            .ok_or_else(|| SLVError::Data("Chart response missing low series".into()))?;
        let closes = quote
            .close
            .ok_or_else(|| SLVError::Data("Chart response missing close series".into()))?;
        let volumes = quote.volume;

        let len = timestamps
            .len()
            .min(opens.len())
            .min(highs.len())
            .min(lows.len())
            .min(closes.len());

        let mut candles = Vec::with_capacity(len);
        for i in 0..len {
            let Some(open) = opens[i] else { continue };
            let Some(high) = highs[i] else { continue };
            let Some(low) = lows[i] else { continue };
            let Some(close) = closes[i] else { continue };
            if !open.is_finite()
                || !high.is_finite()
                || !low.is_finite()
                || !close.is_finite()
                || low > high
            {
                continue;
            }

            let Some(timestamp) = DateTime::from_timestamp(timestamps[i], 0) else {
                continue;
            };

            let volume = volumes
                .as_ref()
                .and_then(|v| v.get(i))
                .and_then(|v| *v)
                .filter(|v| *v >= 0);

            candles.push(CandleBar {
                timestamp,
                open,
                high,
                low,
                close,
                volume: volume.map(|v| v as u64),
            });
        }

        Ok(candles)
    }

    pub fn get_quote(&self, symbol: &str) -> SLVResult<SpotQuote> {
        let url = format!("{}/quote?symbols={}", self.base_url, symbol);

        let response: YahooQuoteResponse = self.get_json_with_crumb(&url, "quote")?;

        let result = response
            .quote_response
            .result
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

    pub fn get_candles(
        &self,
        symbol: &str,
        interval: &str,
        range: &str,
    ) -> SLVResult<Vec<CandleBar>> {
        let mut last_error = None;

        for candidate in Self::options_symbol_candidates(symbol) {
            match self.get_candles_for_symbol(&candidate, interval, range) {
                Ok(candles) if !candles.is_empty() => {
                    if candidate != symbol {
                        tracing::info!(
                            "Using Yahoo chart symbol {} for requested {}",
                            candidate,
                            symbol
                        );
                    }
                    return Ok(candles);
                }
                Ok(_) => {
                    last_error = Some(SLVError::Data(format!(
                        "No candles returned for {}",
                        candidate
                    )));
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| SLVError::Data(format!("No candles available for {}", symbol))))
    }

    pub fn get_expirations(&self, symbol: &str) -> SLVResult<Vec<NaiveDate>> {
        let mut last_error = None;

        for candidate in Self::options_symbol_candidates(symbol) {
            match self.get_expirations_for_symbol(&candidate) {
                Ok(expiries) if !expiries.is_empty() => {
                    if candidate != symbol {
                        tracing::info!(
                            "Using Yahoo options symbol {} for requested {}",
                            candidate,
                            symbol
                        );
                    }
                    return Ok(expiries);
                }
                Ok(_) => {
                    last_error = Some(SLVError::Data(format!(
                        "No expirations returned for {}",
                        candidate
                    )));
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            SLVError::Data(format!("No options expirations available for {}", symbol))
        }))
    }

    pub fn get_option_chain(&self, symbol: &str, expiry: NaiveDate) -> SLVResult<QuoteChain> {
        let mut last_error = None;

        for candidate in Self::options_symbol_candidates(symbol) {
            match self.get_option_chain_for_symbol(symbol, &candidate, expiry) {
                Ok(chain) if !chain.calls.is_empty() || !chain.puts.is_empty() => {
                    if candidate != symbol {
                        tracing::info!(
                            "Using Yahoo options symbol {} for requested {}",
                            candidate,
                            symbol
                        );
                    }
                    return Ok(chain);
                }
                Ok(_) => {
                    last_error = Some(SLVError::Data(format!(
                        "Empty options chain for {} @ {}",
                        candidate, expiry
                    )));
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            SLVError::Data(format!(
                "No option chain returned for {} @ {}",
                symbol, expiry
            ))
        }))
    }

    /// Get full option surface (all expirations)
    pub fn get_option_surface(&self, symbol: &str) -> SLVResult<QuoteSurface> {
        let expiries = self.get_expirations(symbol)?;
        let spot_fallback = self.get_quote(symbol).ok().map(|q| q.price);
        let mut surface = QuoteSurface::new(symbol, spot_fallback.unwrap_or(0.0));

        for expiry in expiries {
            match self.get_option_chain(symbol, expiry) {
                Ok(chain) => {
                    if surface.spot <= 0.0 {
                        surface.spot = chain.spot;
                    }
                    surface.add_chain(chain);
                }
                Err(e) => {
                    tracing::warn!("Failed to get chain for {}: {}", expiry, e);
                }
            }
        }

        if surface.chains.is_empty() {
            return Err(SLVError::Data(format!(
                "No option chains returned for {}",
                symbol
            )));
        }

        Ok(surface)
    }

    pub fn get_option_surface_limited(
        &self,
        symbol: &str,
        max_expiries: usize,
    ) -> SLVResult<QuoteSurface> {
        let expiries = self.get_expirations(symbol)?;
        let spot_fallback = self.get_quote(symbol).ok().map(|q| q.price);
        let mut surface = QuoteSurface::new(symbol, spot_fallback.unwrap_or(0.0));

        for expiry in expiries.into_iter().take(max_expiries) {
            match self.get_option_chain(symbol, expiry) {
                Ok(chain) => {
                    if surface.spot <= 0.0 {
                        surface.spot = chain.spot;
                    }
                    surface.add_chain(chain);
                }
                Err(e) => {
                    tracing::warn!("Failed to get chain for {}: {}", expiry, e);
                }
            }
        }

        if surface.chains.is_empty() {
            return Err(SLVError::Data(format!(
                "No option chains returned for {}",
                symbol
            )));
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
            exercise: ExerciseStyle::American,
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

        quote.compute_mid();

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleBar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: Option<u64>,
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
struct YahooChartResponse {
    chart: YahooChartContainer,
}

#[derive(Debug, Deserialize)]
struct YahooChartContainer {
    result: Option<Vec<YahooChartResult>>,
    error: Option<YahooChartError>,
}

#[derive(Debug, Deserialize)]
struct YahooChartError {
    code: Option<String>,
    description: String,
}

#[derive(Debug, Deserialize)]
struct YahooChartResult {
    timestamp: Option<Vec<i64>>,
    indicators: YahooChartIndicators,
}

#[derive(Debug, Deserialize)]
struct YahooChartIndicators {
    quote: Vec<YahooChartQuote>,
}

#[derive(Debug, Deserialize)]
struct YahooChartQuote {
    open: Option<Vec<Option<f64>>>,
    high: Option<Vec<Option<f64>>>,
    low: Option<Vec<Option<f64>>>,
    close: Option<Vec<Option<f64>>>,
    volume: Option<Vec<Option<i64>>>,
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

            println!(
                "Chain for {}: {} calls, {} puts",
                expiry,
                chain.calls.len(),
                chain.puts.len()
            );

            assert!(!chain.calls.is_empty());
            assert!(!chain.puts.is_empty());
        }
    }

    #[test]
    #[ignore] // Requires network
    fn test_get_candles() {
        let client = YahooClient::new();
        let candles = client.get_candles("SPY", "5m", "5d").unwrap();
        assert!(!candles.is_empty());
    }
}
