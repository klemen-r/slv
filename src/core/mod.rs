//! Core data types for SLV Options Model
//!
//! Defines fundamental types:
//! - OptionContract: Strike, expiry, type (call/put)
//! - OptionQuote: Bid/ask/mid, IV, Greeks
//! - VolSurface: Implied volatility surface
//! - SpotData: Underlying price data

pub mod option;
pub mod quote;
pub mod surface;
pub mod greeks;
pub mod error;

pub use option::*;
pub use quote::*;
pub use surface::*;
pub use greeks::*;
pub use error::*;
