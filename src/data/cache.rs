//! Local data caching
//!
//! Caches option data locally to reduce API calls and enable offline analysis.

use std::path::{Path, PathBuf};
use std::fs;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};

use crate::core::{QuoteSurface, SLVError, SLVResult};

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Maximum age before refresh (in hours)
    pub max_age_hours: i64,
    /// Whether to use cache
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("./data/cache"),
            max_age_hours: 24,
            enabled: true,
        }
    }
}

/// Data cache manager
pub struct DataCache {
    config: CacheConfig,
}

impl DataCache {
    pub fn new(config: CacheConfig) -> SLVResult<Self> {
        // Create cache directory if needed
        if config.enabled && !config.cache_dir.exists() {
            fs::create_dir_all(&config.cache_dir)
                .map_err(|e| SLVError::IO(e))?;
        }

        Ok(Self { config })
    }

    /// Cache key for a symbol and data type
    fn cache_key(&self, symbol: &str, data_type: &str) -> PathBuf {
        self.config.cache_dir.join(format!("{}_{}.json", symbol, data_type))
    }

    /// Check if cache is valid (exists and not expired)
    pub fn is_valid(&self, symbol: &str, data_type: &str) -> bool {
        if !self.config.enabled {
            return false;
        }

        let path = self.cache_key(symbol, data_type);
        if !path.exists() {
            return false;
        }

        // Check modification time
        if let Ok(metadata) = fs::metadata(&path) {
            if let Ok(modified) = metadata.modified() {
                let modified: DateTime<Utc> = modified.into();
                let age = Utc::now() - modified;
                return age < Duration::hours(self.config.max_age_hours);
            }
        }

        false
    }

    /// Save quote surface to cache
    pub fn save_surface(&self, symbol: &str, surface: &QuoteSurface) -> SLVResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let path = self.cache_key(symbol, "surface");
        let json = serde_json::to_string_pretty(surface)
            .map_err(|e| SLVError::Serialization(e.to_string()))?;

        fs::write(&path, json).map_err(|e| SLVError::IO(e))?;

        tracing::info!("Cached surface for {} at {:?}", symbol, path);
        Ok(())
    }

    /// Load quote surface from cache
    pub fn load_surface(&self, symbol: &str) -> SLVResult<Option<QuoteSurface>> {
        if !self.config.enabled || !self.is_valid(symbol, "surface") {
            return Ok(None);
        }

        let path = self.cache_key(symbol, "surface");
        let json = fs::read_to_string(&path).map_err(|e| SLVError::IO(e))?;

        let surface: QuoteSurface = serde_json::from_str(&json)
            .map_err(|e| SLVError::Serialization(e.to_string()))?;

        tracing::info!("Loaded surface for {} from cache", symbol);
        Ok(Some(surface))
    }

    /// Clear cache for a symbol
    pub fn clear(&self, symbol: &str) -> SLVResult<()> {
        let pattern = format!("{}_*.json", symbol);

        for entry in fs::read_dir(&self.config.cache_dir).map_err(|e| SLVError::IO(e))? {
            let entry = entry.map_err(|e| SLVError::IO(e))?;
            let file_name = entry.file_name().to_string_lossy().to_string();

            if file_name.starts_with(&format!("{}_", symbol)) {
                fs::remove_file(entry.path()).map_err(|e| SLVError::IO(e))?;
            }
        }

        Ok(())
    }

    /// Clear all cache
    pub fn clear_all(&self) -> SLVResult<()> {
        if self.config.cache_dir.exists() {
            fs::remove_dir_all(&self.config.cache_dir).map_err(|e| SLVError::IO(e))?;
            fs::create_dir_all(&self.config.cache_dir).map_err(|e| SLVError::IO(e))?;
        }
        Ok(())
    }

    /// List cached symbols
    pub fn list_cached(&self) -> SLVResult<Vec<String>> {
        let mut symbols = Vec::new();

        if !self.config.cache_dir.exists() {
            return Ok(symbols);
        }

        for entry in fs::read_dir(&self.config.cache_dir).map_err(|e| SLVError::IO(e))? {
            let entry = entry.map_err(|e| SLVError::IO(e))?;
            let file_name = entry.file_name().to_string_lossy().to_string();

            if file_name.ends_with("_surface.json") {
                let symbol = file_name.trim_end_matches("_surface.json").to_string();
                if !symbols.contains(&symbol) {
                    symbols.push(symbol);
                }
            }
        }

        Ok(symbols)
    }
}

/// Cached data fetcher - combines cache with live fetching
pub struct CachedFetcher {
    cache: DataCache,
}

impl CachedFetcher {
    pub fn new(config: CacheConfig) -> SLVResult<Self> {
        Ok(Self {
            cache: DataCache::new(config)?,
        })
    }

    /// Get option surface (from cache or fetch)
    pub fn get_surface(&self, symbol: &str) -> SLVResult<QuoteSurface> {
        // Try cache first
        if let Some(surface) = self.cache.load_surface(symbol)? {
            return Ok(surface);
        }

        // Fetch from Yahoo
        tracing::info!("Fetching fresh data for {}", symbol);
        let client = super::yahoo::YahooClient::new();
        let surface = client.get_option_surface(symbol)?;

        // Cache it
        self.cache.save_surface(symbol, &surface)?;

        Ok(surface)
    }

    /// Force refresh (bypass cache)
    pub fn refresh_surface(&self, symbol: &str) -> SLVResult<QuoteSurface> {
        self.cache.clear(symbol)?;
        self.get_surface(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cache_operations() {
        let temp_dir = tempdir().unwrap();
        let config = CacheConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            max_age_hours: 24,
            enabled: true,
        };

        let cache = DataCache::new(config).unwrap();

        // Create a test surface
        let surface = QuoteSurface::new("TEST", 100.0);

        // Save and load
        cache.save_surface("TEST", &surface).unwrap();

        assert!(cache.is_valid("TEST", "surface"));

        let loaded = cache.load_surface("TEST").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().underlying, "TEST");

        // Clear
        cache.clear("TEST").unwrap();
        assert!(!cache.is_valid("TEST", "surface"));
    }
}
