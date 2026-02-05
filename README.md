<p align="center">
  <pre>
 ███████╗██╗    ██╗   ██╗
 ██╔════╝██║    ██║   ██║
 ███████╗██║    ██║   ██║
 ╚════██║██║    ╚██╗ ██╔╝
 ███████║███████╗╚████╔╝ 
 ╚══════╝╚══════╝ ╚═══╝  

 ─────────┼─────────
 ╱╲_╱╲╱╲_╱│╲_╱╲╱╲_╱╲
          ATM
  </pre>
  <h1 align="center">SLV Options</h1>
  <p align="center">
    <strong>Stochastic Local Volatility Model for Options Pricing & Level Detection</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#level-detection">Level Detection</a> •
    <a href="#gui">GUI</a>
  </p>
</p>

---

A production-grade Rust library for options pricing, volatility surface calibration, and **novel level detection** from local volatility surfaces. Identifies statistically significant strikes that influence intraday price behavior.

## Features

### Pricing Models

| Model | Description |
|-------|-------------|
| **Black-Scholes** | Baseline pricing, Greeks, IV solver |
| **Local Volatility** | Dupire model - perfectly fits vanilla smile |
| **Heston** | Stochastic volatility with mean reversion |
| **SLV** | Combined local + stochastic vol with leverage function |

### Level Detection

Detects two types of significant strikes from the local volatility surface:

| Type | Visual | Behavior |
|------|--------|----------|
| **Spike** (Wall/Pivot) | 🔴 Red line | Price tends to reject, pin, or rotate |
| **Air Pocket** (Acceleration) | 🔵 Blue line | Price travels quickly through |

### Data Sources

- **Yahoo Finance** - Equity options (SPY, QQQ, etc.)
- **Sierra Chart** - Futures options (NQ, ES)

---

## Installation

```bash
git clone https://github.com/klemen-r/slv.git
cd slv
cargo build --release
```

---

## Quick Start

### Library Usage

```rust
use slv_options::prelude::*;

// Build volatility surface from market data
let vol_surface = VolSurface::from_grid(
    "SPY", spot, today,
    strikes, times, expiries, iv_grid,
    rate, div_yield,
);

// Compute local volatility (Dupire)
let local_vol = LocalVolSurface::from_implied_vol(&vol_surface)?;

// Detect levels
let levels = detect_levels(&local_vol, spot);

for level in levels.priority_levels() {
    println!("{}", level.tooltip());
    // "K=500 | Wall/Pivot | Confirmed: 1D,4D | Strength: 2.8σ | Distance: +3"
}
```

### Pricing Example

```rust
use slv_options::prelude::*;

let price = bs_price(
    500.0,              // spot
    505.0,              // strike  
    0.25,               // time (years)
    0.05,               // rate
    0.01,               // dividend
    0.20,               // volatility
    OptionType::Call,
);

let greeks = bs_greeks(spot, strike, time, rate, div, vol, OptionType::Call);
println!("Delta: {:.4}, Gamma: {:.4}", greeks.delta, greeks.gamma);
```

---

## Level Detection

### Algorithm Overview

The level detection pipeline has three stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Dislocation Detection                                 │
│  ─────────────────────────────────────────────────────────────  │
│  • Compute robust z-score vs neighborhood (±2 strikes)          │
│  • z = (LV - median) / MAD_scale                                │
│  • Flag candidates where |z| ≥ threshold (default 2.5)          │
│  • Non-maximum suppression to avoid clusters                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Cross-Expiry Confirmation                             │
│  ─────────────────────────────────────────────────────────────  │
│  • Merge candidates across 0D / 1D / 4D expiries                │
│  • Weight: 4D (1.0) > 1D (0.8) > 0D (0.6)                       │
│  • Confidence: High (≥1.8) | Medium (≥1.0) | Low (<1.0)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Proximity Filter                                      │
│  ─────────────────────────────────────────────────────────────  │
│  • Priority band: ±6 strikes from spot (configurable)           │
│  • Far levels shown as "trend magnets" with reduced opacity     │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```rust
let config = LevelConfig {
    dislocation: DislocationConfig {
        window: 2,              // Neighborhood: ±2 strikes
        threshold: 2.5,         // Z-score threshold
        suppression_radius: 1,  // NMS radius
        ..Default::default()
    },
    confirmation: ConfirmationConfig {
        weight_0d: 0.6,
        weight_1d: 0.8,
        weight_4d: 1.0,
        ..Default::default()
    },
    proximity: ProximityConfig {
        priority_band: 6.0,     // Strikes from spot
        ..Default::default()
    },
};

let detector = LevelDetector::with_config(config);
let result = detector.detect(&local_vol_surface, spot);
```

### Output Structure

```rust
pub struct Level {
    pub strike: f64,                    // Strike price
    pub kind: LevelKind,                // Spike or AirPocket
    pub confidence: Confidence,         // High / Medium / Low
    pub score: f64,                     // Overall score
    pub confirm_expiries: Vec<ExpiryBucket>,  // Which expiries confirm
    pub z_by_expiry: ZScoreByExpiry,    // Z-scores per expiry
    pub distance_strikes: f64,          // Distance from spot
    pub priority: bool,                 // Within priority band?
    pub render_style: RenderStyle,      // GUI hints (opacity, width)
}
```

---

## GUI

Interactive visualization of volatility surfaces and detected levels.

```bash
cargo run --bin slv-gui
```

### Features

- **Blue curve** — Implied volatility smile
- **Green curve** — Local volatility (Dupire)
- **Red/orange lines** — Spike levels (Walls/Pivots)
- **Cyan/blue lines** — Air pocket levels (Acceleration)
- **Yellow dashed** — ATM (spot price)
- **Configurable** threshold and proximity band
- **Levels table** with strike, type, confidence, distance

### Controls

| Control | Function |
|---------|----------|
| **Fetch** | Download options data from Yahoo Finance |
| **Load** | Load cached data |
| **Threshold** | Z-score threshold (lower = more levels) |
| **Proximity** | Priority band size in strikes |
| **Expiry** | Select expiry to analyze |

---

## Project Structure

```
slv-options/
├── src/
│   ├── lib.rs                 # Library entry
│   ├── core/                  # Core types
│   │   ├── option.rs          # OptionContract, OptionType
│   │   ├── quote.rs           # OptionQuote, QuoteChain
│   │   ├── surface.rs         # VolSurface
│   │   ├── greeks.rs          # Greeks
│   │   └── error.rs           # Error types
│   ├── models/                # Pricing models
│   │   ├── black_scholes.rs   # BS pricing & IV
│   │   ├── local_vol.rs       # Dupire local vol
│   │   ├── heston.rs          # Heston stochastic vol
│   │   └── slv.rs             # Combined SLV
│   ├── levels/                # Level detection
│   │   ├── mod.rs             # Types (Level, LevelKind, etc.)
│   │   ├── config.rs          # Configuration
│   │   ├── detection.rs       # Stage 1: Dislocation
│   │   ├── confirmation.rs    # Stage 2: Cross-expiry
│   │   ├── proximity.rs       # Stage 3: Proximity
│   │   └── detector.rs        # LevelDetector facade
│   ├── data/                  # Data fetching
│   │   ├── yahoo.rs           # Yahoo Finance
│   │   └── sierra.rs          # Sierra Chart
│   └── bin/                   # Executables
│       ├── slv_gui.rs         # Interactive GUI
│       └── slv_calibrate.rs   # Calibration CLI
├── examples/
│   ├── basic_pricing.rs       # Pricing example
│   └── detect_levels.rs       # Level detection example
└── scripts/
    └── fetch_options.py       # Yahoo data fetcher
```

---

## Examples

### Run Examples

```bash
# Basic Black-Scholes pricing
cargo run --example basic_pricing

# Level detection
cargo run --example detect_levels
```

### Level Detection Output

```
=== Level Detection Results ===

Spot: 500.00
Strike spacing: 1.00
Total levels: 2

--- Detected Levels ---

Strike 500: SPIKE | High | z=3.20 | dist=+0.0 strikes | PRIORITY
Strike 505: AIR_POCKET | High | z=2.80 | dist=+5.0 strikes | PRIORITY

--- Summary ---

Spikes (Walls/Pivots): 1
Air Pockets (Acceleration): 1
High confidence: 2
Priority (within band): 2
```

---

## API Reference

### Key Types

| Type | Description |
|------|-------------|
| `Level` | Detected level with all metadata |
| `LevelKind` | `Spike` or `AirPocket` |
| `Confidence` | `High`, `Medium`, or `Low` |
| `LevelDetector` | Main detection facade |
| `LevelConfig` | Configuration for all stages |
| `VolSurface` | Implied volatility surface |
| `LocalVolSurface` | Dupire local vol surface |

### Key Functions

| Function | Description |
|----------|-------------|
| `detect_levels(lv_surface, spot)` | Detect levels with default config |
| `detect_levels_with_config(...)` | Detect with custom config |
| `bs_price(...)` | Black-Scholes price |
| `bs_greeks(...)` | Black-Scholes Greeks |
| `implied_volatility(...)` | IV solver |

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `ndarray` | N-dimensional arrays |
| `statrs` | Statistical functions |
| `num-complex` | Complex numbers (Heston) |
| `eframe` / `egui` | GUI framework |
| `reqwest` | HTTP client |
| `serde` | Serialization |

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for educational and research purposes. It does **not**:
- Predict future volatility or prices
- Generate trading signals
- Account for market microstructure
- Handle American exercise optimally

Use at your own risk.
