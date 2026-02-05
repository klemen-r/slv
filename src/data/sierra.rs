//! Sierra Chart data integration
//!
//! Reads NQ futures options data from Sierra Chart SCID files
//! and provides integration with the Denali data feed.

use std::path::Path;
use std::fs::File;
use std::io::{BufReader, Read};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use crate::core::{SLVError, SLVResult};

/// Sierra Chart SCID file reader for futures options
pub struct SierraOptionsReader {
    data_dir: String,
}

impl SierraOptionsReader {
    pub fn new(data_dir: impl Into<String>) -> Self {
        Self {
            data_dir: data_dir.into(),
        }
    }

    /// Parse NQ futures option symbol
    /// Format: OQN[Month][Year][C/P][Strike]-CME
    /// Example: OQNH25C21500-CME (March 2025 Call at 21500)
    pub fn parse_option_symbol(symbol: &str) -> Option<FuturesOptionInfo> {
        // Remove exchange suffix
        let symbol = symbol.trim_end_matches("-CME");

        if !symbol.starts_with("OQN") {
            return None;
        }

        let rest = &symbol[3..];
        if rest.len() < 6 {
            return None;
        }

        // Month code (H=Mar, M=Jun, U=Sep, Z=Dec for NQ)
        let month_code = rest.chars().next()?;
        let month = match month_code {
            'H' => 3,
            'M' => 6,
            'U' => 9,
            'Z' => 12,
            _ => return None,
        };

        // Year (2 digits)
        let year: i32 = rest[1..3].parse().ok()?;
        let year = 2000 + year;

        // Option type
        let opt_type_char = rest.chars().nth(3)?;
        let is_call = match opt_type_char {
            'C' => true,
            'P' => false,
            _ => return None,
        };

        // Strike price
        let strike: f64 = rest[4..].parse().ok()?;

        // Compute expiry (3rd Friday of expiry month, typically)
        let expiry = compute_futures_option_expiry(year, month)?;

        Some(FuturesOptionInfo {
            underlying: "NQ".to_string(),
            expiry,
            strike,
            is_call,
            symbol: symbol.to_string(),
        })
    }

    /// List available option SCID files in data directory
    pub fn list_option_files(&self) -> SLVResult<Vec<String>> {
        let path = Path::new(&self.data_dir);

        if !path.exists() {
            return Err(SLVError::Data(format!(
                "Data directory does not exist: {}", self.data_dir
            )));
        }

        let mut files = Vec::new();

        for entry in std::fs::read_dir(path)
            .map_err(|e| SLVError::IO(e))?
        {
            let entry = entry.map_err(|e| SLVError::IO(e))?;
            let file_name = entry.file_name().to_string_lossy().to_string();

            // Match NQ option files: OQN*.scid
            if file_name.starts_with("OQN") && file_name.ends_with(".scid") {
                files.push(file_name);
            }
        }

        Ok(files)
    }

    /// Read last price from SCID file
    pub fn read_last_price(&self, symbol: &str) -> SLVResult<Option<f64>> {
        let file_path = Path::new(&self.data_dir).join(format!("{}.scid", symbol));

        if !file_path.exists() {
            return Ok(None);
        }

        // SCID format: Read the last record
        let file = File::open(&file_path).map_err(|e| SLVError::IO(e))?;
        let metadata = file.metadata().map_err(|e| SLVError::IO(e))?;
        let file_size = metadata.len();

        if file_size < 56 + 40 {
            // Header (56 bytes) + at least one record (40 bytes)
            return Ok(None);
        }

        let mut reader = BufReader::new(file);

        // Skip to last record
        let record_size = 40u64;
        let num_records = (file_size - 56) / record_size;
        if num_records == 0 {
            return Ok(None);
        }

        // Seek to last record
        use std::io::Seek;
        reader.seek(std::io::SeekFrom::Start(56 + (num_records - 1) * record_size))
            .map_err(|e| SLVError::IO(e))?;

        // Read SCID record
        let mut buffer = [0u8; 40];
        reader.read_exact(&mut buffer).map_err(|e| SLVError::IO(e))?;

        // Parse close price (bytes 24-31, f64 little endian)
        let close = f64::from_le_bytes(buffer[24..32].try_into().unwrap());

        Ok(Some(close))
    }

    /// Get available NQ option chains
    pub fn get_available_chains(&self) -> SLVResult<Vec<FuturesOptionChain>> {
        let files = self.list_option_files()?;

        let mut chains: std::collections::HashMap<NaiveDate, FuturesOptionChain> =
            std::collections::HashMap::new();

        for file in files {
            let symbol = file.trim_end_matches(".scid");

            if let Some(info) = Self::parse_option_symbol(symbol) {
                let chain = chains.entry(info.expiry).or_insert_with(|| {
                    FuturesOptionChain::new("NQ", info.expiry)
                });

                if let Ok(Some(price)) = self.read_last_price(symbol) {
                    chain.add_option(info.strike, info.is_call, price);
                }
            }
        }

        let mut result: Vec<_> = chains.into_values().collect();
        result.sort_by_key(|c| c.expiry);

        Ok(result)
    }
}

/// Parsed futures option info
#[derive(Debug, Clone)]
pub struct FuturesOptionInfo {
    pub underlying: String,
    pub expiry: NaiveDate,
    pub strike: f64,
    pub is_call: bool,
    pub symbol: String,
}

/// Futures option chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturesOptionChain {
    pub underlying: String,
    pub expiry: NaiveDate,
    pub strikes: Vec<f64>,
    pub calls: std::collections::HashMap<i64, f64>, // strike (as cents) -> price
    pub puts: std::collections::HashMap<i64, f64>,
    pub timestamp: DateTime<Utc>,
}

impl FuturesOptionChain {
    pub fn new(underlying: &str, expiry: NaiveDate) -> Self {
        Self {
            underlying: underlying.to_string(),
            expiry,
            strikes: Vec::new(),
            calls: std::collections::HashMap::new(),
            puts: std::collections::HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn add_option(&mut self, strike: f64, is_call: bool, price: f64) {
        let strike_key = (strike * 100.0) as i64;

        if !self.strikes.contains(&strike) {
            self.strikes.push(strike);
            self.strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }

        if is_call {
            self.calls.insert(strike_key, price);
        } else {
            self.puts.insert(strike_key, price);
        }
    }

    pub fn call_price(&self, strike: f64) -> Option<f64> {
        let key = (strike * 100.0) as i64;
        self.calls.get(&key).copied()
    }

    pub fn put_price(&self, strike: f64) -> Option<f64> {
        let key = (strike * 100.0) as i64;
        self.puts.get(&key).copied()
    }
}

/// Compute futures option expiry date
/// NQ options expire on the 3rd Friday of the contract month
fn compute_futures_option_expiry(year: i32, month: u32) -> Option<NaiveDate> {
    // Find first day of month
    let first = NaiveDate::from_ymd_opt(year, month, 1)?;

    // Friday is day 4 (Monday=0, Tuesday=1, ..., Friday=4)
    // If first is Monday (0), days to Friday = 4
    // If first is Tuesday (1), days to Friday = 3
    // If first is Saturday (5), days to Friday = 6
    // If first is Sunday (6), days to Friday = 5
    let first_weekday = first.weekday().num_days_from_monday();
    let days_to_friday = (4 + 7 - first_weekday) % 7;

    // If first day is already Friday, days_to_friday = 0, which is correct
    let first_friday = first + chrono::Duration::days(days_to_friday as i64);

    // Third Friday is 14 days later
    let third_friday = first_friday + chrono::Duration::days(14);

    Some(third_friday)
}

/// Index for NDX cash index from Sierra Chart
#[derive(Debug, Clone)]
pub struct NDXIndexReader {
    data_dir: String,
}

impl NDXIndexReader {
    pub fn new(data_dir: impl Into<String>) -> Self {
        Self {
            data_dir: data_dir.into(),
        }
    }

    /// Read NDX index value from CBOE Global Indexes
    /// Symbol: $NDX-CBOE or NDX_CGI-CBOE
    pub fn read_last_value(&self) -> SLVResult<Option<f64>> {
        // Try different symbol formats
        let symbols = ["$NDX-CBOE.scid", "NDX_CGI-CBOE.scid", "NDX.scid"];

        for symbol in symbols {
            let file_path = Path::new(&self.data_dir).join(symbol);
            if file_path.exists() {
                // Use same SCID reading logic
                let reader = SierraOptionsReader::new(&self.data_dir);
                return reader.read_last_price(symbol.trim_end_matches(".scid"));
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_option_symbol() {
        let info = SierraOptionsReader::parse_option_symbol("OQNH25C21500-CME").unwrap();

        assert_eq!(info.underlying, "NQ");
        assert_eq!(info.strike, 21500.0);
        assert!(info.is_call);
        assert_eq!(info.expiry.month(), 3);
        assert_eq!(info.expiry.year(), 2025);
    }

    #[test]
    fn test_compute_expiry() {
        // March 2025 - 3rd Friday should be March 21, 2025
        let expiry = compute_futures_option_expiry(2025, 3).unwrap();
        assert_eq!(expiry.day(), 21);
    }
}
