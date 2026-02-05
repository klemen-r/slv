//! Level Derivation from Local Volatility Surface
//!
//! Detects statistically significant strikes ("levels") that influence intraday price behavior.
//!
//! Two level types:
//! - **Spikes (Walls/Pivots)**: Elevated local vol → price tends to reject, pin, or rotate
//! - **Air-pockets (Acceleration)**: Depressed local vol → price travels quickly ("fast-through")
//!
//! Three-stage pipeline:
//! 1. **Dislocation detection**: Robust z-score vs neighborhood, non-maximum suppression
//! 2. **Cross-expiry confirmation**: Merge candidates across 0D/1D/4D, compute confidence
//! 3. **Proximity filter**: Prioritize levels within intraday reach of spot

mod config;
mod confirmation;
mod detection;
mod detector;
mod proximity;

pub use config::*;
pub use confirmation::*;
pub use detection::*;
pub use detector::*;
pub use proximity::*;

use serde::{Deserialize, Serialize};

/// Type of detected level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LevelKind {
    /// Elevated local vol - price tends to reject/pin/rotate
    Spike,
    /// Depressed local vol - price tends to travel quickly through
    AirPocket,
}

impl LevelKind {
    /// User-friendly label for GUI
    pub fn label(&self) -> &'static str {
        match self {
            LevelKind::Spike => "Wall / Pivot",
            LevelKind::AirPocket => "Acceleration",
        }
    }

    /// Short label
    pub fn short_label(&self) -> &'static str {
        match self {
            LevelKind::Spike => "SPIKE",
            LevelKind::AirPocket => "AIR_POCKET",
        }
    }
}

/// Confidence tier for a level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Confidence {
    Low,
    Medium,
    High,
}

impl Confidence {
    /// From confirmation score (sum of expiry weights)
    pub fn from_score(s_confirm: f64) -> Self {
        if s_confirm >= 1.8 {
            Confidence::High
        } else if s_confirm >= 1.0 {
            Confidence::Medium
        } else {
            Confidence::Low
        }
    }

    /// Default opacity for rendering
    pub fn opacity(&self) -> f32 {
        match self {
            Confidence::High => 1.0,
            Confidence::Medium => 0.7,
            Confidence::Low => 0.35,
        }
    }
}

/// Expiry bucket for cross-expiry analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpiryBucket {
    /// 0 days to expiry (same-day)
    ZeroD,
    /// 1 day to expiry
    OneD,
    /// ~4 days to expiry (weekly)
    FourD,
}

impl ExpiryBucket {
    /// Default weight for confirmation scoring
    pub fn default_weight(&self) -> f64 {
        match self {
            ExpiryBucket::ZeroD => 0.6,
            ExpiryBucket::OneD => 0.8,
            ExpiryBucket::FourD => 1.0,
        }
    }

    /// Label for display
    pub fn label(&self) -> &'static str {
        match self {
            ExpiryBucket::ZeroD => "0D",
            ExpiryBucket::OneD => "1D",
            ExpiryBucket::FourD => "4D",
        }
    }

    /// Classify time-to-expiry (in days) into bucket
    pub fn from_days(days: f64) -> Option<Self> {
        if days < 0.5 {
            Some(ExpiryBucket::ZeroD)
        } else if days < 1.5 {
            Some(ExpiryBucket::OneD)
        } else if days < 5.0 {
            Some(ExpiryBucket::FourD)
        } else {
            None // Too far out for intraday relevance
        }
    }
}

/// A candidate level from Stage 1 (single expiry)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelCandidate {
    /// Strike price
    pub strike: f64,
    /// Strike index in the grid
    pub strike_idx: usize,
    /// Robust z-score (positive = spike, negative = air-pocket)
    pub z_score: f64,
    /// Level type
    pub kind: LevelKind,
    /// Source expiry bucket
    pub expiry: ExpiryBucket,
    /// Local vol value at this strike
    pub local_vol: f64,
    /// Neighborhood median (baseline)
    pub baseline: f64,
}

/// Per-expiry z-score for a confirmed level
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZScoreByExpiry {
    pub zero_d: Option<f64>,
    pub one_d: Option<f64>,
    pub four_d: Option<f64>,
}

impl ZScoreByExpiry {
    pub fn get(&self, expiry: ExpiryBucket) -> Option<f64> {
        match expiry {
            ExpiryBucket::ZeroD => self.zero_d,
            ExpiryBucket::OneD => self.one_d,
            ExpiryBucket::FourD => self.four_d,
        }
    }

    pub fn set(&mut self, expiry: ExpiryBucket, z: f64) {
        match expiry {
            ExpiryBucket::ZeroD => self.zero_d = Some(z),
            ExpiryBucket::OneD => self.one_d = Some(z),
            ExpiryBucket::FourD => self.four_d = Some(z),
        }
    }

    /// Format for tooltip: "1D: 3.1σ, 4D: 2.7σ"
    pub fn tooltip(&self) -> String {
        let mut parts = Vec::new();
        if let Some(z) = self.zero_d {
            parts.push(format!("0D: {:.1}σ", z));
        }
        if let Some(z) = self.one_d {
            parts.push(format!("1D: {:.1}σ", z));
        }
        if let Some(z) = self.four_d {
            parts.push(format!("4D: {:.1}σ", z));
        }
        parts.join(", ")
    }
}

/// Render style hints for GUI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderStyle {
    /// Opacity (0.0 to 1.0)
    pub opacity: f32,
    /// Line weight multiplier
    pub line_weight: f32,
    /// Whether this is a priority level (within proximity band)
    pub priority: bool,
}

impl RenderStyle {
    pub fn from_level(confidence: Confidence, priority: bool) -> Self {
        let base_opacity = confidence.opacity();
        let opacity = if priority {
            base_opacity
        } else {
            base_opacity * 0.5
        };
        let line_weight = match confidence {
            Confidence::High => 2.0,
            Confidence::Medium => 1.5,
            Confidence::Low => 1.0,
        };

        Self {
            opacity,
            line_weight,
            priority,
        }
    }
}

/// Final detected level (output of full pipeline)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level {
    /// Strike price (snapped to grid)
    pub strike: f64,
    /// Level type
    pub kind: LevelKind,
    /// Confidence tier
    pub confidence: Confidence,
    /// Overall score (0.0 to ~2.4)
    pub score: f64,
    /// Expiries that confirm this level
    pub confirm_expiries: Vec<ExpiryBucket>,
    /// Z-score per expiry
    pub z_by_expiry: ZScoreByExpiry,
    /// Distance from spot in strikes (positive = above, negative = below)
    pub distance_strikes: f64,
    /// Whether within priority proximity band
    pub priority: bool,
    /// Render hints for GUI
    pub render_style: RenderStyle,
}

impl Level {
    /// Generate tooltip text for GUI
    pub fn tooltip(&self) -> String {
        let expiries: Vec<&str> = self.confirm_expiries.iter().map(|e| e.label()).collect();
        format!(
            "K={:.0} | {} | Confirmed: {} | Strength: {} | Distance: {:+.0}",
            self.strike,
            self.kind.label(),
            expiries.join(","),
            self.z_by_expiry.tooltip(),
            self.distance_strikes
        )
    }

    /// Max z-score across all confirming expiries
    pub fn max_z_score(&self) -> f64 {
        [
            self.z_by_expiry.zero_d,
            self.z_by_expiry.one_d,
            self.z_by_expiry.four_d,
        ]
        .into_iter()
        .flatten()
        .map(|z| z.abs())
        .fold(0.0, f64::max)
    }
}

/// Result of level detection pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelDetectionResult {
    /// Detected levels, sorted by distance from spot
    pub levels: Vec<Level>,
    /// Spot price used
    pub spot: f64,
    /// Strike spacing (median diff)
    pub strike_spacing: f64,
    /// Configuration used
    pub config: LevelConfig,
    /// Number of candidates before confirmation
    pub candidates_per_expiry: std::collections::HashMap<String, usize>,
}

impl LevelDetectionResult {
    /// Get levels within N strikes of spot
    pub fn levels_within(&self, n_strikes: f64) -> Vec<&Level> {
        self.levels
            .iter()
            .filter(|l| l.distance_strikes.abs() <= n_strikes)
            .collect()
    }

    /// Get only high-confidence levels
    pub fn high_confidence(&self) -> Vec<&Level> {
        self.levels
            .iter()
            .filter(|l| l.confidence == Confidence::High)
            .collect()
    }

    /// Get spikes only
    pub fn spikes(&self) -> Vec<&Level> {
        self.levels
            .iter()
            .filter(|l| l.kind == LevelKind::Spike)
            .collect()
    }

    /// Get air-pockets only
    pub fn air_pockets(&self) -> Vec<&Level> {
        self.levels
            .iter()
            .filter(|l| l.kind == LevelKind::AirPocket)
            .collect()
    }

    /// Priority levels (within proximity band)
    pub fn priority_levels(&self) -> Vec<&Level> {
        self.levels.iter().filter(|l| l.priority).collect()
    }
}
