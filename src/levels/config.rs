//! Configuration for level detection pipeline

use super::ExpiryBucket;
use serde::{Deserialize, Serialize};

/// Configuration for level detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelConfig {
    /// Stage 1: Dislocation detection
    pub dislocation: DislocationConfig,
    /// Stage 2: Cross-expiry confirmation
    pub confirmation: ConfirmationConfig,
    /// Stage 3: Proximity filter
    pub proximity: ProximityConfig,
}

impl Default for LevelConfig {
    fn default() -> Self {
        Self {
            dislocation: DislocationConfig::default(),
            confirmation: ConfirmationConfig::default(),
            proximity: ProximityConfig::default(),
        }
    }
}

impl LevelConfig {
    /// Aggressive settings: lower thresholds, more levels
    pub fn aggressive() -> Self {
        Self {
            dislocation: DislocationConfig {
                threshold: 2.0,
                ..Default::default()
            },
            confirmation: ConfirmationConfig {
                high_confidence_threshold: 1.5,
                medium_confidence_threshold: 0.8,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Conservative settings: higher thresholds, fewer levels
    pub fn conservative() -> Self {
        Self {
            dislocation: DislocationConfig {
                threshold: 3.0,
                ..Default::default()
            },
            confirmation: ConfirmationConfig {
                high_confidence_threshold: 2.0,
                medium_confidence_threshold: 1.2,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Stage 1: Dislocation detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DislocationConfig {
    /// Window size (strikes on each side, excluding center)
    /// Default: 2 (5-point neighborhood)
    pub window: usize,

    /// Z-score threshold for candidate detection
    /// |z| >= threshold marks a candidate
    /// Default: 2.5
    pub threshold: f64,

    /// Non-maximum suppression radius (strikes)
    /// Suppress weaker candidates within ±d strikes
    /// Default: 1
    pub suppression_radius: usize,

    /// Minimum scale (MAD) to avoid division by near-zero
    /// Default: 1e-6
    pub epsilon: f64,

    /// Maximum z-score to cap outliers in magnitude scoring
    /// Default: 5.0
    pub z_max: f64,
}

impl Default for DislocationConfig {
    fn default() -> Self {
        Self {
            window: 2,
            threshold: 2.5,
            suppression_radius: 1,
            epsilon: 1e-6,
            z_max: 5.0,
        }
    }
}

/// Stage 2: Cross-expiry confirmation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfirmationConfig {
    /// Match tolerance for strikes across expiries
    /// Strikes match if |Ka - Kb| <= tolerance
    /// Default: 1.0 (one strike step)
    pub match_tolerance: f64,

    /// Weight for 0D expiry
    pub weight_0d: f64,
    /// Weight for 1D expiry
    pub weight_1d: f64,
    /// Weight for 4D expiry
    pub weight_4d: f64,

    /// Threshold for high confidence (S_confirm >= this)
    /// Default: 1.8
    pub high_confidence_threshold: f64,

    /// Threshold for medium confidence (S_confirm >= this)
    /// Default: 1.0
    pub medium_confidence_threshold: f64,

    /// Weight for confirmation score in final score
    /// Default: 0.5
    pub confirm_weight: f64,

    /// Weight for magnitude score in final score
    /// Default: 0.5
    pub magnitude_weight: f64,

    /// Whether to include low-confidence levels in output
    /// Default: false
    pub include_low_confidence: bool,
}

impl Default for ConfirmationConfig {
    fn default() -> Self {
        Self {
            match_tolerance: 1.0,
            weight_0d: 0.6,
            weight_1d: 0.8,
            weight_4d: 1.0,
            high_confidence_threshold: 1.8,
            medium_confidence_threshold: 1.0,
            confirm_weight: 0.5,
            magnitude_weight: 0.5,
            include_low_confidence: false,
        }
    }
}

impl ConfirmationConfig {
    /// Get weight for an expiry bucket
    pub fn weight(&self, expiry: ExpiryBucket) -> f64 {
        match expiry {
            ExpiryBucket::ZeroD => self.weight_0d,
            ExpiryBucket::OneD => self.weight_1d,
            ExpiryBucket::FourD => self.weight_4d,
        }
    }
}

/// Stage 3: Proximity filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximityConfig {
    /// Priority band in strikes from spot
    /// Levels within ±B strikes are marked as priority
    /// Default: 6
    pub priority_band: f64,

    /// Maximum distance to include at all (in strikes)
    /// Levels beyond this are excluded
    /// Default: 20 (or None for no limit)
    pub max_distance: Option<f64>,

    /// Secondary threshold for "trend magnet" levels
    /// Far levels become visible when price approaches within this
    /// Default: 3
    pub magnet_approach_threshold: f64,
}

impl Default for ProximityConfig {
    fn default() -> Self {
        Self {
            priority_band: 6.0,
            max_distance: Some(20.0),
            magnet_approach_threshold: 3.0,
        }
    }
}
