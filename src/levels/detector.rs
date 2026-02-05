//! LevelDetector - Main facade for the level detection pipeline
//!
//! Combines all three stages into a single interface.

use std::collections::HashMap;

use crate::models::LocalVolSurface;

use super::{
    apply_proximity_filter, compute_strike_spacing, detect_dislocations, merge_and_confirm,
    ExpiryBucket, LevelCandidate, LevelConfig, LevelDetectionResult,
};

/// Main level detector that runs the full three-stage pipeline
pub struct LevelDetector {
    config: LevelConfig,
}

impl LevelDetector {
    /// Create a new detector with default configuration
    pub fn new() -> Self {
        Self {
            config: LevelConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: LevelConfig) -> Self {
        Self { config }
    }

    /// Get current configuration
    pub fn config(&self) -> &LevelConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: LevelConfig) {
        self.config = config;
    }

    /// Run full detection pipeline on a local volatility surface
    ///
    /// # Arguments
    /// * `lv_surface` - Local volatility surface from Dupire
    /// * `spot` - Current spot price
    ///
    /// # Returns
    /// Detection result with all levels
    pub fn detect(&self, lv_surface: &LocalVolSurface, spot: f64) -> LevelDetectionResult {
        let strikes = &lv_surface.log_moneyness;

        // Convert log-moneyness to absolute strikes for output
        // log_moneyness = ln(K/F), so K = F * exp(log_moneyness)
        // For simplicity, we'll use spot as proxy for forward
        let absolute_strikes: Vec<f64> = strikes.iter().map(|&y| spot * y.exp()).collect();

        // Compute strike spacing
        let strike_spacing = compute_strike_spacing(&absolute_strikes);

        // Stage 1: Detect dislocations per expiry
        let mut candidates_by_expiry: HashMap<ExpiryBucket, Vec<LevelCandidate>> = HashMap::new();
        let mut candidates_count: HashMap<String, usize> = HashMap::new();

        for expiry_bucket in [ExpiryBucket::ZeroD, ExpiryBucket::OneD, ExpiryBucket::FourD] {
            if let Some(local_vols) = self.extract_local_vol_smile(lv_surface, expiry_bucket) {
                let candidates = detect_dislocations(
                    &absolute_strikes,
                    &local_vols,
                    expiry_bucket,
                    &self.config.dislocation,
                );

                candidates_count.insert(expiry_bucket.label().to_string(), candidates.len());

                if !candidates.is_empty() {
                    candidates_by_expiry.insert(expiry_bucket, candidates);
                }
            }
        }

        // Stage 2: Cross-expiry confirmation
        let clusters = merge_and_confirm(candidates_by_expiry, &self.config.confirmation);

        // Stage 3: Proximity filter
        let levels = apply_proximity_filter(clusters, spot, strike_spacing, &self.config.proximity);

        LevelDetectionResult {
            levels,
            spot,
            strike_spacing,
            config: self.config.clone(),
            candidates_per_expiry: candidates_count,
        }
    }

    /// Detect levels from raw data (strikes, times, local_var grid)
    ///
    /// # Arguments
    /// * `strikes` - Absolute strike prices
    /// * `times` - Times to expiry in years
    /// * `local_var` - Local variance grid [strike, time] (σ²_local)
    /// * `spot` - Current spot price
    pub fn detect_from_grid(
        &self,
        strikes: &[f64],
        times: &[f64],
        local_var: &ndarray::Array2<f64>,
        spot: f64,
    ) -> LevelDetectionResult {
        let strike_spacing = compute_strike_spacing(strikes);

        // Stage 1: Detect dislocations per expiry
        let mut candidates_by_expiry: HashMap<ExpiryBucket, Vec<LevelCandidate>> = HashMap::new();
        let mut candidates_count: HashMap<String, usize> = HashMap::new();

        for (ti, &time) in times.iter().enumerate() {
            let days = time * 365.0;
            if let Some(expiry_bucket) = ExpiryBucket::from_days(days) {
                // Extract local vol column for this expiry
                let local_vols: Vec<f64> = (0..strikes.len())
                    .map(|si| local_var[[si, ti]].sqrt())
                    .collect();

                let candidates = detect_dislocations(
                    strikes,
                    &local_vols,
                    expiry_bucket,
                    &self.config.dislocation,
                );

                candidates_count.insert(expiry_bucket.label().to_string(), candidates.len());

                if !candidates.is_empty() {
                    // Merge with existing candidates for this bucket (if multiple expiries map to same bucket)
                    candidates_by_expiry
                        .entry(expiry_bucket)
                        .or_insert_with(Vec::new)
                        .extend(candidates);
                }
            }
        }

        // Stage 2: Cross-expiry confirmation
        let clusters = merge_and_confirm(candidates_by_expiry, &self.config.confirmation);

        // Stage 3: Proximity filter
        let levels = apply_proximity_filter(clusters, spot, strike_spacing, &self.config.proximity);

        LevelDetectionResult {
            levels,
            spot,
            strike_spacing,
            config: self.config.clone(),
            candidates_per_expiry: candidates_count,
        }
    }

    /// Extract local vol smile for a specific expiry bucket
    fn extract_local_vol_smile(
        &self,
        lv_surface: &LocalVolSurface,
        target: ExpiryBucket,
    ) -> Option<Vec<f64>> {
        let target_days = match target {
            ExpiryBucket::ZeroD => 0.0,
            ExpiryBucket::OneD => 1.0,
            ExpiryBucket::FourD => 4.0,
        };
        let target_years = target_days / 365.0;

        // Find closest time
        let (ti, &closest_time) =
            lv_surface
                .times
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (*a - target_years).abs();
                    let db = (*b - target_years).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })?;

        // Check if it's reasonably close
        let actual_days = closest_time * 365.0;
        let bucket = ExpiryBucket::from_days(actual_days)?;
        if bucket != target {
            return None;
        }

        // Extract local vols (sqrt of variance)
        let local_vols: Vec<f64> = (0..lv_surface.log_moneyness.len())
            .map(|si| lv_surface.local_var[[si, ti]].sqrt())
            .collect();

        Some(local_vols)
    }
}

impl Default for LevelDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to detect levels from a LocalVolSurface
pub fn detect_levels(lv_surface: &LocalVolSurface, spot: f64) -> LevelDetectionResult {
    LevelDetector::new().detect(lv_surface, spot)
}

/// Convenience function with custom config
pub fn detect_levels_with_config(
    lv_surface: &LocalVolSurface,
    spot: f64,
    config: LevelConfig,
) -> LevelDetectionResult {
    LevelDetector::with_config(config).detect(lv_surface, spot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_local_var_grid() -> (Vec<f64>, Vec<f64>, Array2<f64>) {
        // Create a grid with some deliberate spikes and air pockets
        let strikes: Vec<f64> = (0..21).map(|i| 95.0 + i as f64 * 0.5).collect(); // 95 to 105
        let times = vec![0.0027, 0.0055, 0.011]; // ~1d, ~2d, ~4d in years

        let n_strikes = strikes.len();
        let n_times = times.len();
        let mut local_var = Array2::from_elem((n_strikes, n_times), 0.04); // Base: 20% vol → 0.04 variance

        // Add a spike at strike index 10 (strike = 100) across all expiries
        for ti in 0..n_times {
            local_var[[10, ti]] = 0.09; // 30% vol
        }

        // Add an air pocket at strike index 15 (strike = 102.5) in 1D and 4D
        local_var[[15, 0]] = 0.04; // Normal in 0D
        local_var[[15, 1]] = 0.01; // 10% vol - air pocket in 1D
        local_var[[15, 2]] = 0.01; // 10% vol - air pocket in 4D

        (strikes, times, local_var)
    }

    #[test]
    fn test_full_pipeline() {
        let (strikes, times, local_var) = create_test_local_var_grid();
        let spot = 100.0;

        let detector = LevelDetector::new();
        let result = detector.detect_from_grid(&strikes, &times, &local_var, spot);

        // Should detect at least the spike
        assert!(
            !result.levels.is_empty(),
            "Should detect at least one level"
        );

        // Check that we have spike detection
        let spikes: Vec<_> = result.spikes();
        assert!(!spikes.is_empty(), "Should detect the spike");

        // The spike should be near strike 100
        let spike = spikes
            .iter()
            .min_by(|a, b| {
                (a.strike - 100.0)
                    .abs()
                    .partial_cmp(&(b.strike - 100.0).abs())
                    .unwrap()
            })
            .unwrap();
        assert!(
            (spike.strike - 100.0).abs() < 2.0,
            "Spike should be near 100, got {}",
            spike.strike
        );
    }

    #[test]
    fn test_priority_levels() {
        let (strikes, times, local_var) = create_test_local_var_grid();
        let spot = 100.0;

        let detector = LevelDetector::new();
        let result = detector.detect_from_grid(&strikes, &times, &local_var, spot);

        // Priority levels should be within the proximity band (default 6 strikes)
        for level in result.priority_levels() {
            assert!(
                level.distance_strikes.abs() <= 6.0,
                "Priority level should be within band"
            );
        }
    }

    #[test]
    fn test_conservative_config() {
        let (strikes, times, local_var) = create_test_local_var_grid();
        let spot = 100.0;

        let conservative = LevelDetector::with_config(LevelConfig::conservative());
        let aggressive = LevelDetector::with_config(LevelConfig::aggressive());

        let result_conservative = conservative.detect_from_grid(&strikes, &times, &local_var, spot);
        let result_aggressive = aggressive.detect_from_grid(&strikes, &times, &local_var, spot);

        // Conservative should have fewer or equal levels
        assert!(
            result_conservative.levels.len() <= result_aggressive.levels.len(),
            "Conservative config should produce fewer levels"
        );
    }
}
