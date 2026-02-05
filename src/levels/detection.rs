//! Stage 1: Dislocation Detection
//!
//! Detects local-vol anomalies using robust z-scores and non-maximum suppression.

use super::{DislocationConfig, ExpiryBucket, LevelCandidate, LevelKind};

/// Detect dislocations in a local volatility smile for a single expiry
///
/// # Arguments
/// * `strikes` - Strike prices (sorted ascending)
/// * `local_vols` - Local volatility values at each strike
/// * `expiry` - Which expiry bucket this is
/// * `config` - Detection configuration
///
/// # Returns
/// Vector of candidate levels after non-maximum suppression
pub fn detect_dislocations(
    strikes: &[f64],
    local_vols: &[f64],
    expiry: ExpiryBucket,
    config: &DislocationConfig,
) -> Vec<LevelCandidate> {
    let n = strikes.len();
    if n < 2 * config.window + 1 {
        return Vec::new(); // Not enough data for neighborhood
    }

    // Step 1: Compute robust z-scores for all strikes
    let z_scores = compute_robust_z_scores(local_vols, config);

    // Step 2: Find candidates exceeding threshold
    let mut candidates: Vec<LevelCandidate> = Vec::new();

    for i in config.window..(n - config.window) {
        let z = z_scores[i];
        let abs_z = z.abs();

        if abs_z >= config.threshold {
            let kind = if z > 0.0 {
                LevelKind::Spike
            } else {
                LevelKind::AirPocket
            };

            // Compute baseline for this strike
            let (baseline, _) = compute_neighborhood_stats(local_vols, i, config.window);

            candidates.push(LevelCandidate {
                strike: strikes[i],
                strike_idx: i,
                z_score: z,
                kind,
                expiry,
                local_vol: local_vols[i],
                baseline,
            });
        }
    }

    // Step 3: Non-maximum suppression
    non_maximum_suppression(candidates, config.suppression_radius)
}

/// Compute robust z-scores for all points
///
/// z_i = (LV(K_i) - median(neighborhood)) / max(MAD_scale, epsilon)
fn compute_robust_z_scores(local_vols: &[f64], config: &DislocationConfig) -> Vec<f64> {
    let n = local_vols.len();
    let mut z_scores = vec![0.0; n];

    for i in config.window..(n - config.window) {
        let (median, mad_scale) = compute_neighborhood_stats(local_vols, i, config.window);
        let scale = mad_scale.max(config.epsilon);
        z_scores[i] = (local_vols[i] - median) / scale;
    }

    z_scores
}

/// Compute neighborhood median and MAD scale
///
/// # Returns
/// (median, MAD_scale) where MAD_scale = 1.4826 * median(|x - median|)
fn compute_neighborhood_stats(values: &[f64], center: usize, window: usize) -> (f64, f64) {
    // Collect neighborhood values (excluding center)
    let mut neighborhood: Vec<f64> = Vec::with_capacity(2 * window);

    for offset in 1..=window {
        if center >= offset {
            neighborhood.push(values[center - offset]);
        }
        if center + offset < values.len() {
            neighborhood.push(values[center + offset]);
        }
    }

    if neighborhood.is_empty() {
        return (values[center], 1.0);
    }

    // Compute median
    neighborhood.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if neighborhood.len() % 2 == 0 {
        let mid = neighborhood.len() / 2;
        (neighborhood[mid - 1] + neighborhood[mid]) / 2.0
    } else {
        neighborhood[neighborhood.len() / 2]
    };

    // Compute MAD (Median Absolute Deviation)
    let mut abs_devs: Vec<f64> = neighborhood.iter().map(|&x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mad = if abs_devs.len() % 2 == 0 {
        let mid = abs_devs.len() / 2;
        (abs_devs[mid - 1] + abs_devs[mid]) / 2.0
    } else {
        abs_devs[abs_devs.len() / 2]
    };

    // Scale factor 1.4826 makes MAD consistent with std dev for normal distribution
    let mad_scale = 1.4826 * mad;

    (median, mad_scale)
}

/// Non-maximum suppression to avoid cluster pollution
///
/// Sort by |z| descending, accept strongest, suppress within ±radius
fn non_maximum_suppression(
    mut candidates: Vec<LevelCandidate>,
    radius: usize,
) -> Vec<LevelCandidate> {
    if candidates.is_empty() {
        return candidates;
    }

    // Sort by |z| descending
    candidates.sort_by(|a, b| {
        b.z_score
            .abs()
            .partial_cmp(&a.z_score.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut accepted: Vec<LevelCandidate> = Vec::new();
    let mut suppressed_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for candidate in candidates {
        if suppressed_indices.contains(&candidate.strike_idx) {
            continue;
        }

        // Accept this candidate
        accepted.push(candidate.clone());

        // Suppress neighbors
        let idx = candidate.strike_idx;
        for offset in 1..=radius {
            if idx >= offset {
                suppressed_indices.insert(idx - offset);
            }
            suppressed_indices.insert(idx + offset);
        }
    }

    accepted
}

/// Extract local vol smile at a specific expiry bucket from the surface
///
/// Returns (strikes, local_vols) for the closest matching expiry
pub fn extract_smile_for_expiry(
    strikes: &[f64],
    times: &[f64],
    local_vol_grid: &ndarray::Array2<f64>,
    target_expiry: ExpiryBucket,
) -> Option<(Vec<f64>, Vec<f64>)> {
    // Find time index matching the expiry bucket
    let target_days = match target_expiry {
        ExpiryBucket::ZeroD => 0.0,
        ExpiryBucket::OneD => 1.0,
        ExpiryBucket::FourD => 4.0,
    };
    let target_years = target_days / 365.0;

    // Find closest time
    let (ti, _) = times.iter().enumerate().min_by(|(_, a), (_, b)| {
        let da = (*a - target_years).abs();
        let db = (*b - target_years).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })?;

    // Check if it's reasonably close
    let actual_days = times[ti] * 365.0;
    let bucket = ExpiryBucket::from_days(actual_days)?;
    if bucket != target_expiry {
        return None;
    }

    // Extract local vols for this expiry
    let local_vols: Vec<f64> = (0..strikes.len())
        .map(|si| {
            // local_vol_grid is [strike, time]
            local_vol_grid[[si, ti]].sqrt() // Convert variance to vol
        })
        .collect();

    Some((strikes.to_vec(), local_vols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighborhood_stats() {
        // Simple case: [1, 2, 3, 4, 5]
        let values = vec![1.0, 2.0, 10.0, 4.0, 5.0]; // 10 is an outlier
        let (median, mad_scale) = compute_neighborhood_stats(&values, 2, 2);

        // Neighborhood: [1, 2, 4, 5], median = 3.0
        assert!((median - 3.0).abs() < 0.01);
        // MAD of [1, 2, 4, 5] around 3: |1-3|=2, |2-3|=1, |4-3|=1, |5-3|=2
        // Sorted: [1, 1, 2, 2], median = 1.5, MAD_scale = 1.4826 * 1.5 = 2.22
        assert!((mad_scale - 2.22).abs() < 0.1);
    }

    #[test]
    fn test_spike_detection() {
        // Create a smile with one spike
        let strikes: Vec<f64> = (0..11).map(|i| 100.0 + i as f64).collect();
        let mut local_vols = vec![0.20; 11];
        local_vols[5] = 0.35; // Spike at strike 105

        let config = DislocationConfig::default();
        let candidates = detect_dislocations(&strikes, &local_vols, ExpiryBucket::OneD, &config);

        assert!(!candidates.is_empty());
        let spike = &candidates[0];
        assert_eq!(spike.kind, LevelKind::Spike);
        assert!((spike.strike - 105.0).abs() < 0.01);
        assert!(spike.z_score > 0.0);
    }

    #[test]
    fn test_air_pocket_detection() {
        // Create a smile with one air pocket
        let strikes: Vec<f64> = (0..11).map(|i| 100.0 + i as f64).collect();
        let mut local_vols = vec![0.25; 11];
        local_vols[5] = 0.12; // Air pocket at strike 105

        let config = DislocationConfig::default();
        let candidates = detect_dislocations(&strikes, &local_vols, ExpiryBucket::OneD, &config);

        assert!(!candidates.is_empty());
        let pocket = &candidates[0];
        assert_eq!(pocket.kind, LevelKind::AirPocket);
        assert!((pocket.strike - 105.0).abs() < 0.01);
        assert!(pocket.z_score < 0.0);
    }

    #[test]
    fn test_nms_suppresses_neighbors() {
        // Two adjacent spikes - should keep only the stronger one
        let strikes: Vec<f64> = (0..11).map(|i| 100.0 + i as f64).collect();
        let mut local_vols = vec![0.20; 11];
        local_vols[5] = 0.35; // Stronger spike
        local_vols[6] = 0.30; // Weaker spike

        let config = DislocationConfig {
            suppression_radius: 1,
            ..Default::default()
        };
        let candidates = detect_dislocations(&strikes, &local_vols, ExpiryBucket::OneD, &config);

        // Should only have one candidate (the stronger one at 105)
        assert_eq!(candidates.len(), 1);
        assert!((candidates[0].strike - 105.0).abs() < 0.01);
    }
}
