//! Stage 2: Cross-Expiry Confirmation
//!
//! Merges candidates across expiry buckets and computes confidence scores.

use super::{
    Confidence, ConfirmationConfig, ExpiryBucket, LevelCandidate, LevelKind, ZScoreByExpiry,
};
use std::collections::HashMap;

/// Merged cluster of candidates across expiries
#[derive(Debug, Clone)]
pub struct MergedCluster {
    /// Representative strike (weighted average)
    pub strike: f64,
    /// Level kind (determined by majority or strongest)
    pub kind: LevelKind,
    /// Contributing candidates by expiry
    pub candidates: HashMap<ExpiryBucket, LevelCandidate>,
    /// Confirmation score (sum of weights for present expiries)
    pub s_confirm: f64,
    /// Magnitude score (weighted sum of clamped |z|)
    pub s_magnitude: f64,
    /// Overall score
    pub score: f64,
    /// Confidence tier
    pub confidence: Confidence,
    /// Z-scores by expiry
    pub z_by_expiry: ZScoreByExpiry,
}

impl MergedCluster {
    /// Get list of confirming expiries
    pub fn confirm_expiries(&self) -> Vec<ExpiryBucket> {
        self.candidates.keys().copied().collect()
    }
}

/// Merge candidates across expiries and compute confirmation scores
///
/// # Arguments
/// * `candidates_by_expiry` - Map from expiry bucket to candidates
/// * `config` - Confirmation configuration
///
/// # Returns
/// Vector of merged clusters, filtered by confidence
pub fn merge_and_confirm(
    candidates_by_expiry: HashMap<ExpiryBucket, Vec<LevelCandidate>>,
    config: &ConfirmationConfig,
) -> Vec<MergedCluster> {
    // Collect all candidates with their expiry
    let mut all_candidates: Vec<(ExpiryBucket, LevelCandidate)> = Vec::new();
    for (expiry, candidates) in &candidates_by_expiry {
        for c in candidates {
            all_candidates.push((*expiry, c.clone()));
        }
    }

    if all_candidates.is_empty() {
        return Vec::new();
    }

    // Sort by strike
    all_candidates.sort_by(|a, b| {
        a.1.strike
            .partial_cmp(&b.1.strike)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Cluster candidates within tolerance
    let mut clusters: Vec<MergedCluster> = Vec::new();

    for (expiry, candidate) in all_candidates {
        // Try to merge with existing cluster
        let mut merged = false;
        for cluster in &mut clusters {
            if (cluster.strike - candidate.strike).abs() <= config.match_tolerance {
                // Add to this cluster
                cluster.candidates.insert(expiry, candidate.clone());
                merged = true;
                break;
            }
        }

        if !merged {
            // Create new cluster
            let mut candidates = HashMap::new();
            candidates.insert(expiry, candidate.clone());
            clusters.push(MergedCluster {
                strike: candidate.strike,
                kind: candidate.kind,
                candidates,
                s_confirm: 0.0,
                s_magnitude: 0.0,
                score: 0.0,
                confidence: Confidence::Low,
                z_by_expiry: ZScoreByExpiry::default(),
            });
        }
    }

    // Compute scores for each cluster
    let max_magnitude = compute_max_possible_magnitude(config);

    for cluster in &mut clusters {
        // Update representative strike (weighted by |z|)
        let (weighted_strike, total_weight) = cluster
            .candidates
            .values()
            .map(|c| (c.strike * c.z_score.abs(), c.z_score.abs()))
            .fold((0.0, 0.0), |(ws, tw), (s, w)| (ws + s, tw + w));

        if total_weight > 0.0 {
            cluster.strike = weighted_strike / total_weight;
        }

        // Determine kind by strongest candidate
        let strongest = cluster.candidates.values().max_by(|a, b| {
            a.z_score
                .abs()
                .partial_cmp(&b.z_score.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(c) = strongest {
            cluster.kind = c.kind;
        }

        // Compute confirmation score
        cluster.s_confirm = cluster.candidates.keys().map(|e| config.weight(*e)).sum();

        // Compute magnitude score
        cluster.s_magnitude = cluster
            .candidates
            .iter()
            .map(|(e, c)| {
                let clamped_z = c.z_score.abs().min(5.0); // z_max hardcoded for now
                config.weight(*e) * clamped_z
            })
            .sum();

        // Normalize magnitude score
        let norm_magnitude = if max_magnitude > 0.0 {
            cluster.s_magnitude / max_magnitude
        } else {
            0.0
        };

        // Overall score
        cluster.score =
            config.confirm_weight * cluster.s_confirm + config.magnitude_weight * norm_magnitude;

        // Confidence tier
        cluster.confidence = if cluster.s_confirm >= config.high_confidence_threshold {
            Confidence::High
        } else if cluster.s_confirm >= config.medium_confidence_threshold {
            Confidence::Medium
        } else {
            Confidence::Low
        };

        // Build z_by_expiry
        for (expiry, candidate) in &cluster.candidates {
            cluster.z_by_expiry.set(*expiry, candidate.z_score);
        }
    }

    // Filter by confidence
    if config.include_low_confidence {
        clusters
    } else {
        clusters
            .into_iter()
            .filter(|c| c.confidence >= Confidence::Medium)
            .collect()
    }
}

/// Compute maximum possible magnitude score for normalization
fn compute_max_possible_magnitude(config: &ConfirmationConfig) -> f64 {
    // Max is when all expiries present at z_max
    let z_max = 5.0;
    (config.weight_0d + config.weight_1d + config.weight_4d) * z_max
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(strike: f64, z: f64, expiry: ExpiryBucket) -> LevelCandidate {
        LevelCandidate {
            strike,
            strike_idx: 0,
            z_score: z,
            kind: if z > 0.0 {
                LevelKind::Spike
            } else {
                LevelKind::AirPocket
            },
            expiry,
            local_vol: 0.25,
            baseline: 0.20,
        }
    }

    #[test]
    fn test_single_expiry_low_confidence() {
        let mut candidates_by_expiry = HashMap::new();
        candidates_by_expiry.insert(
            ExpiryBucket::ZeroD,
            vec![make_candidate(100.0, 3.0, ExpiryBucket::ZeroD)],
        );

        let config = ConfirmationConfig::default();
        let clusters = merge_and_confirm(candidates_by_expiry, &config);

        // Only 0D present → s_confirm = 0.6 < 1.0 → Low confidence → filtered out
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_multi_expiry_high_confidence() {
        let mut candidates_by_expiry = HashMap::new();
        candidates_by_expiry.insert(
            ExpiryBucket::OneD,
            vec![make_candidate(100.0, 3.0, ExpiryBucket::OneD)],
        );
        candidates_by_expiry.insert(
            ExpiryBucket::FourD,
            vec![make_candidate(100.0, 2.8, ExpiryBucket::FourD)],
        );

        let config = ConfirmationConfig::default();
        let clusters = merge_and_confirm(candidates_by_expiry, &config);

        // 1D (0.8) + 4D (1.0) = 1.8 → High confidence
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].confidence, Confidence::High);
        assert_eq!(clusters[0].candidates.len(), 2);
    }

    #[test]
    fn test_nearby_strikes_merge() {
        let mut candidates_by_expiry = HashMap::new();
        candidates_by_expiry.insert(
            ExpiryBucket::OneD,
            vec![make_candidate(100.0, 3.0, ExpiryBucket::OneD)],
        );
        candidates_by_expiry.insert(
            ExpiryBucket::FourD,
            vec![make_candidate(100.5, 2.8, ExpiryBucket::FourD)], // Within tolerance of 1.0
        );

        let config = ConfirmationConfig::default();
        let clusters = merge_and_confirm(candidates_by_expiry, &config);

        // Should merge into one cluster
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].candidates.len(), 2);
    }

    #[test]
    fn test_far_strikes_separate() {
        // Test that strikes beyond tolerance form separate clusters
        // With default config: 1D weight = 0.8, 4D weight = 1.0
        // Both are >= medium_confidence_threshold (1.0) so both pass
        let mut candidates_by_expiry = HashMap::new();
        candidates_by_expiry.insert(
            ExpiryBucket::OneD,
            vec![make_candidate(100.0, 3.0, ExpiryBucket::OneD)],
        );
        candidates_by_expiry.insert(
            ExpiryBucket::FourD,
            vec![make_candidate(105.0, 2.8, ExpiryBucket::FourD)], // Beyond tolerance of 1.0
        );

        let config = ConfirmationConfig::default();
        let clusters = merge_and_confirm(candidates_by_expiry, &config);

        // They should form 2 separate clusters
        // 1D alone: s_confirm = 0.8 < 1.0 → filtered (Low confidence)
        // 4D alone: s_confirm = 1.0 >= 1.0 → passes (Medium confidence)
        // So only one cluster should remain
        assert_eq!(clusters.len(), 1);
        assert!((clusters[0].strike - 105.0).abs() < 0.01); // The 4D one
    }
}
