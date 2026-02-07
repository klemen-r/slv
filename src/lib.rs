//! # SLV Options - Stochastic Local Volatility Model
//!
//! A production-grade options pricing library implementing the Stochastic Local
//! Volatility (SLV) model for NDX/NQ options.
//!
//! ## Overview
//!
//! The SLV model combines:
//! - **Local Volatility (Dupire)**: Perfectly fits the vanilla smile
//! - **Heston Stochastic Volatility**: Realistic forward smile dynamics
//!
//! The result is a model that both fits market prices exactly AND produces
//! sensible dynamics for exotic pricing and hedging.
//!
//! ## Key Components
//!
//! - **Data Fetching**: Yahoo Finance (QQQ) and Sierra Chart (NQ futures options)
//! - **Black-Scholes**: Baseline pricing and IV solver
//! - **Local Vol**: Dupire local volatility surface
//! - **Heston**: Stochastic volatility via characteristic function
//! - **SLV**: Combined model with leverage function
//!
//! ## Usage
//!
//! ```rust,no_run
//! use slv_options::prelude::*;
//!
//! // Fetch QQQ options from Yahoo Finance
//! let surface = fetch_qqq_options().unwrap();
//!
//! // Compute IV surface
//! let iv_surface = compute_iv_surface(&surface);
//!
//! // Build SLV model
//! let params = SLVParams::typical_equity(0.5);
//! let model = SLVModel::from_iv_surface(&iv_surface, params).unwrap();
//!
//! // Price options
//! let call_price = model.mc_price(500.0, 0.25, OptionType::Call, 10000, 100);
//! ```
//!
//! ## What This Model Does
//!
//! - Calibrates to market vanilla option prices
//! - Produces realistic smile dynamics
//! - Supports Monte Carlo simulation
//! - Computes Greeks via finite differences
//!
//! ## What This Model Does NOT Do
//!
//! - Predict future volatility or prices
//! - Generate trading signals
//! - Account for market microstructure
//! - Handle American exercise optimally (uses European approximation)

pub mod calibration;
pub mod core;
pub mod data;
pub mod levels;
pub mod models;
pub mod pricing;

/// Prelude with commonly used types
pub mod prelude {
    // Core types
    pub use crate::core::{
        ExerciseStyle, Greeks, OptionContract, OptionQuote, OptionType, PortfolioGreeks,
        QuoteChain, QuoteSurface, SLVError, SLVResult, SmileParams, VolGridType, VolSurface,
    };

    // Data fetching
    pub use crate::data::{
        fetch_qqq_options, fetch_spy_options, CacheConfig, CachedFetcher, CandleBar, DataCache,
        FuturesOptionChain, SierraOptionsReader, SpotQuote, YahooClient,
    };

    // Models
    pub use crate::models::{
        calibrate_slv,
        compute_iv_surface,
        greeks as bs_greeks,
        implied_volatility,
        norm_cdf,
        norm_pdf,

        // Black-Scholes
        price as bs_price,
        HestonModel,

        // Heston
        HestonParams,
        LeverageFunction,
        LocalVolModel,

        // Local Vol
        LocalVolSurface,
        SLVCalibrationResult,
        SLVModel,
        // SLV
        SLVParams,
    };

    // Level Detection
    pub use crate::levels::{
        detect_levels,
        detect_levels_with_config,
        Confidence,
        ConfirmationConfig,
        DislocationConfig,
        ExpiryBucket,
        // Core types
        Level,
        LevelCandidate,
        // Config
        LevelConfig,
        LevelDetectionResult,
        // Detector
        LevelDetector,
        LevelKind,
        ProximityConfig,
        RenderStyle,
        ZScoreByExpiry,
    };
}

// Re-export main types at crate root
pub use crate::core::{SLVError, SLVResult};
pub use crate::models::{HestonParams, SLVModel, SLVParams};
