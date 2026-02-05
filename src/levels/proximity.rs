//! Stage 3: Proximity Filter
//!
//! Filters and prioritizes levels based on distance from spot.

use super::{Level, MergedCluster, ProximityConfig, RenderStyle};

/// Apply proximity filter and produce final levels
///
/// # Arguments
/// * `clusters` - Confirmed clusters from Stage 2
/// * `spot` - Current spot price
/// * `strike_spacing` - Median strike spacing (for distance calculation)
/// * `config` - Proximity configuration
///
/// # Returns
/// Final levels sorted by distance from spot
pub fn apply_proximity_filter(
    clusters: Vec<MergedCluster>,
    spot: f64,
    strike_spacing: f64,
    config: &ProximityConfig,
) -> Vec<Level> {
    let mut levels: Vec<Level> = Vec::new();

    for cluster in clusters {
        // Compute distance in strikes
        let distance_strikes = if strike_spacing > 0.0 {
            (cluster.strike - spot) / strike_spacing
        } else {
            cluster.strike - spot
        };

        // Check max distance filter
        if let Some(max_dist) = config.max_distance {
            if distance_strikes.abs() > max_dist {
                continue;
            }
        }

        // Determine priority
        let priority = distance_strikes.abs() <= config.priority_band;

        // Build render style
        let render_style = RenderStyle::from_level(cluster.confidence, priority);

        levels.push(Level {
            strike: cluster.strike,
            kind: cluster.kind,
            confidence: cluster.confidence,
            score: cluster.score,
            confirm_expiries: cluster.confirm_expiries(),
            z_by_expiry: cluster.z_by_expiry,
            distance_strikes,
            priority,
            render_style,
        });
    }

    // Sort by distance from spot (closest first)
    levels.sort_by(|a, b| {
        a.distance_strikes
            .abs()
            .partial_cmp(&b.distance_strikes.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    levels
}

/// Compute median strike spacing from a sorted strike array
pub fn compute_strike_spacing(strikes: &[f64]) -> f64 {
    if strikes.len() < 2 {
        return 1.0;
    }

    let mut diffs: Vec<f64> = strikes
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .filter(|&d| d > 1e-10) // Filter out duplicates
        .collect();

    if diffs.is_empty() {
        return 1.0;
    }

    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Return median
    if diffs.len() % 2 == 0 {
        let mid = diffs.len() / 2;
        (diffs[mid - 1] + diffs[mid]) / 2.0
    } else {
        diffs[diffs.len() / 2]
    }
}

/// Update level priorities based on current spot
///
/// Call this when spot moves to re-evaluate which levels are in the priority band
pub fn update_priorities(
    levels: &mut [Level],
    spot: f64,
    strike_spacing: f64,
    config: &ProximityConfig,
) {
    for level in levels.iter_mut() {
        // Recompute distance
        level.distance_strikes = if strike_spacing > 0.0 {
            (level.strike - spot) / strike_spacing
        } else {
            level.strike - spot
        };

        // Update priority
        level.priority = level.distance_strikes.abs() <= config.priority_band;

        // Update render style
        level.render_style = RenderStyle::from_level(level.confidence, level.priority);
    }

    // Re-sort by distance
    levels.sort_by(|a, b| {
        a.distance_strikes
            .abs()
            .partial_cmp(&b.distance_strikes.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Check if a "trend magnet" level should become visible
///
/// Returns true if the level is within the approach threshold of spot
pub fn is_magnet_approaching(level: &Level, config: &ProximityConfig) -> bool {
    !level.priority
        && level.distance_strikes.abs() <= config.magnet_approach_threshold + config.priority_band
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::{Confidence, ExpiryBucket, LevelKind, ZScoreByExpiry};
    use std::collections::HashMap;

    fn make_cluster(strike: f64, s_confirm: f64, kind: LevelKind) -> MergedCluster {
        let confidence = Confidence::from_score(s_confirm);
        let mut candidates = HashMap::new();
        candidates.insert(
            ExpiryBucket::OneD,
            crate::levels::LevelCandidate {
                strike,
                strike_idx: 0,
                z_score: 3.0,
                kind,
                expiry: ExpiryBucket::OneD,
                local_vol: 0.25,
                baseline: 0.20,
            },
        );
        candidates.insert(
            ExpiryBucket::FourD,
            crate::levels::LevelCandidate {
                strike,
                strike_idx: 0,
                z_score: 2.8,
                kind,
                expiry: ExpiryBucket::FourD,
                local_vol: 0.25,
                baseline: 0.20,
            },
        );

        MergedCluster {
            strike,
            kind,
            candidates,
            s_confirm,
            s_magnitude: 5.0,
            score: 1.5,
            confidence,
            z_by_expiry: ZScoreByExpiry {
                zero_d: None,
                one_d: Some(3.0),
                four_d: Some(2.8),
            },
        }
    }

    #[test]
    fn test_proximity_filter() {
        let clusters = vec![
            make_cluster(100.0, 1.8, LevelKind::Spike), // At spot
            make_cluster(105.0, 1.8, LevelKind::Spike), // 5 strikes away
            make_cluster(108.0, 1.8, LevelKind::Spike), // 8 strikes away (outside priority)
            make_cluster(125.0, 1.8, LevelKind::Spike), // 25 strikes away (outside max)
        ];

        let config = ProximityConfig {
            priority_band: 6.0,
            max_distance: Some(20.0),
            magnet_approach_threshold: 3.0,
        };

        let levels = apply_proximity_filter(clusters, 100.0, 1.0, &config);

        // Should have 3 levels (one filtered by max_distance)
        assert_eq!(levels.len(), 3);

        // First two should be priority
        assert!(levels[0].priority); // 100.0
        assert!(levels[1].priority); // 105.0
        assert!(!levels[2].priority); // 108.0

        // Should be sorted by distance
        assert!(levels[0].distance_strikes.abs() <= levels[1].distance_strikes.abs());
        assert!(levels[1].distance_strikes.abs() <= levels[2].distance_strikes.abs());
    }

    #[test]
    fn test_strike_spacing() {
        let strikes = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let spacing = compute_strike_spacing(&strikes);
        assert!((spacing - 1.0).abs() < 0.01);

        let strikes2 = vec![100.0, 105.0, 110.0, 115.0];
        let spacing2 = compute_strike_spacing(&strikes2);
        assert!((spacing2 - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_update_priorities() {
        let clusters = vec![
            make_cluster(100.0, 1.8, LevelKind::Spike),
            make_cluster(110.0, 1.8, LevelKind::Spike),
        ];

        let config = ProximityConfig::default();
        let mut levels = apply_proximity_filter(clusters, 100.0, 1.0, &config);

        // Initially at spot=100: 100 is priority, 110 is not (distance=10 > 6)
        assert!(
            levels
                .iter()
                .find(|l| (l.strike - 100.0).abs() < 0.1)
                .unwrap()
                .priority
        );
        assert!(
            !levels
                .iter()
                .find(|l| (l.strike - 110.0).abs() < 0.1)
                .unwrap()
                .priority
        );

        // Move spot to 108
        update_priorities(&mut levels, 108.0, 1.0, &config);

        // Now 110 should be priority (distance=2), 100 might not be (distance=8)
        assert!(
            levels
                .iter()
                .find(|l| (l.strike - 110.0).abs() < 0.1)
                .unwrap()
                .priority
        );
        assert!(
            !levels
                .iter()
                .find(|l| (l.strike - 100.0).abs() < 0.1)
                .unwrap()
                .priority
        );
    }
}
