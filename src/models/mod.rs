//! Volatility Models
//!
//! Implements:
//! - Black-Scholes (baseline pricing, IV computation)
//! - Local Volatility (Dupire)
//! - Heston Stochastic Volatility
//! - Stochastic Local Volatility (SLV) - the main model

pub mod black_scholes;
pub mod local_vol;
pub mod heston;
pub mod slv;

pub use black_scholes::*;
pub use local_vol::*;
pub use heston::*;
pub use slv::*;
