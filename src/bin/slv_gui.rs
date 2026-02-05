//! SLV Options GUI
//!
//! Interactive visualization of volatility surfaces, SLV model, and detected levels.

use chrono::NaiveDate;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, VLine};
use ndarray::Array2;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::process::Command;

use slv_options::prelude::*;

#[derive(Debug, Deserialize)]
struct OptionsData {
    symbol: String,
    spot: f64,
    #[allow(dead_code)]
    timestamp: String,
    chains: Vec<ChainData>,
}

#[derive(Debug, Deserialize)]
struct ChainData {
    expiry: String,
    calls: Vec<OptionData>,
    #[allow(dead_code)]
    puts: Vec<OptionData>,
}

#[derive(Debug, Deserialize)]
struct OptionData {
    strike: f64,
    #[allow(dead_code)]
    bid: Option<f64>,
    #[allow(dead_code)]
    ask: Option<f64>,
    #[allow(dead_code)]
    last: Option<f64>,
    iv: Option<f64>,
    #[allow(dead_code)]
    volume: i64,
    #[allow(dead_code)]
    open_interest: i64,
}

struct SLVApp {
    // Data
    symbol: String,
    spot: f64,
    strikes: Vec<f64>,
    times: Vec<f64>,
    iv_grid: Array2<f64>,

    // UI state
    selected_expiry: usize,
    show_calls: bool,
    show_local_vol: bool,
    show_levels: bool,
    show_level_labels: bool,

    // Ticker fetch
    ticker_input: String,
    num_expiries: u32,
    fetch_status: String,
    is_fetching: bool,

    // Level detection config
    level_threshold: f64,
    level_proximity_band: f64,

    // Model params
    heston_v0: f64,
    heston_kappa: f64,
    heston_theta: f64,
    heston_sigma: f64,
    heston_rho: f64,
    slv_alpha: f64,

    // Computed
    local_vol_surface: Option<LocalVolSurface>,
    vol_surface: Option<VolSurface>,
    detected_levels: Option<LevelDetectionResult>,
}

impl Default for SLVApp {
    fn default() -> Self {
        Self {
            symbol: "SPY".to_string(),
            spot: 500.0,
            strikes: vec![],
            times: vec![],
            iv_grid: Array2::zeros((0, 0)),
            selected_expiry: 0,
            show_calls: true,
            show_local_vol: true,
            show_levels: true,
            show_level_labels: true,
            ticker_input: "SPY".to_string(),
            num_expiries: 6,
            fetch_status: String::new(),
            is_fetching: false,
            level_threshold: 2.5,
            level_proximity_band: 6.0,
            heston_v0: 0.04,
            heston_kappa: 2.0,
            heston_theta: 0.04,
            heston_sigma: 0.3,
            heston_rho: -0.7,
            slv_alpha: 0.5,
            local_vol_surface: None,
            vol_surface: None,
            detected_levels: None,
        }
    }
}

impl SLVApp {
    fn fetch_options(&mut self) {
        let ticker = self.ticker_input.trim().to_uppercase();
        if ticker.is_empty() {
            self.fetch_status = "Enter a ticker symbol".to_string();
            return;
        }

        self.fetch_status = format!("Fetching {}...", ticker);
        self.is_fetching = true;

        let script_path = "D:/Stochastic local volatility/slv-options/scripts/fetch_options.py";
        let result = Command::new("python")
            .arg(script_path)
            .arg(&ticker)
            .arg(self.num_expiries.to_string())
            .output();

        match result {
            Ok(output) => {
                if output.status.success() {
                    self.fetch_status = format!("{} data fetched!", ticker);
                    self.ticker_input = ticker;
                    self.load_data();
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    self.fetch_status =
                        format!("Error: {}", stderr.lines().next().unwrap_or("Failed"));
                }
            }
            Err(e) => {
                self.fetch_status = format!("Error: {}", e);
            }
        }
        self.is_fetching = false;
    }

    fn load_data(&mut self) {
        let ticker = self.ticker_input.trim().to_lowercase();
        let data_path = format!(
            "D:/Stochastic local volatility/slv-options/data/{}_options.json",
            ticker
        );

        let file = match File::open(data_path) {
            Ok(f) => f,
            Err(_) => return,
        };

        let reader = BufReader::new(file);
        let data: OptionsData = match serde_json::from_reader(reader) {
            Ok(d) => d,
            Err(_) => return,
        };

        self.symbol = data.symbol;
        self.spot = data.spot;

        let today = chrono::Utc::now().date_naive();
        let rate = 0.045;
        let div = 0.013;

        // Build surface
        let mut strikes: Vec<f64> = Vec::new();
        let mut times: Vec<f64> = Vec::new();
        let mut expiries: Vec<NaiveDate> = Vec::new();

        for chain in &data.chains {
            let expiry = NaiveDate::parse_from_str(&chain.expiry, "%Y-%m-%d").unwrap();
            let tte = (expiry - today).num_days() as f64 / 365.25;
            if tte <= 0.0 {
                continue;
            }

            times.push(tte);
            expiries.push(expiry);

            for call in &chain.calls {
                if !strikes.contains(&call.strike) {
                    strikes.push(call.strike);
                }
            }
        }

        strikes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Filter to reasonable range
        let atm_idx = strikes
            .iter()
            .position(|&k| k >= self.spot)
            .unwrap_or(strikes.len() / 2);
        let start = atm_idx.saturating_sub(20);
        let end = (atm_idx + 20).min(strikes.len());
        strikes = strikes[start..end].to_vec();

        // Build IV grid
        let mut iv_grid: Array2<f64> = Array2::zeros((strikes.len(), times.len()));

        let mut ti = 0;
        for chain in data.chains.iter() {
            let expiry = NaiveDate::parse_from_str(&chain.expiry, "%Y-%m-%d").unwrap();
            let tte = (expiry - today).num_days() as f64 / 365.25;
            if tte <= 0.0 {
                continue;
            }

            for call in &chain.calls {
                if let Some(si) = strikes.iter().position(|&k| (k - call.strike).abs() < 0.01) {
                    if let Some(iv) = call.iv {
                        if iv > 0.01 && iv < 3.0 {
                            iv_grid[[si, ti]] = iv;
                        }
                    }
                }
            }
            ti += 1;
        }

        // Fill gaps
        for ti in 0..times.len() {
            let mut last_valid = None;
            for si in 0..strikes.len() {
                if iv_grid[[si, ti]] > 0.0 {
                    last_valid = Some(iv_grid[[si, ti]]);
                } else if let Some(last_iv) = last_valid {
                    iv_grid[[si, ti]] = last_iv;
                }
            }
            let mut last_valid = None;
            for si in (0..strikes.len()).rev() {
                if iv_grid[[si, ti]] > 0.0 {
                    last_valid = Some(iv_grid[[si, ti]]);
                } else if let Some(last_iv) = last_valid {
                    iv_grid[[si, ti]] = last_iv;
                }
            }
        }

        self.strikes = strikes.clone();
        self.times = times.clone();
        self.iv_grid = iv_grid.clone();

        // Build vol surface
        let vol_surface = VolSurface::from_grid(
            &self.symbol,
            self.spot,
            today,
            strikes,
            times,
            expiries,
            iv_grid,
            rate,
            div,
        );

        // Build local vol
        if let Ok(lv) = LocalVolSurface::from_implied_vol(&vol_surface) {
            self.local_vol_surface = Some(lv);
        }

        self.vol_surface = Some(vol_surface);

        // Detect levels
        self.detect_levels();
    }

    fn detect_levels(&mut self) {
        if self.strikes.is_empty() || self.times.is_empty() {
            self.detected_levels = None;
            return;
        }

        let mut config = LevelConfig::default();
        config.dislocation.threshold = self.level_threshold;
        config.proximity.priority_band = self.level_proximity_band;

        // Try full pipeline using local vol surface (cross-expiry confirmation)
        if let Some(ref lv_surface) = self.local_vol_surface {
            let detector = LevelDetector::with_config(config.clone());
            let result = detector.detect(lv_surface, self.spot);
            if !result.levels.is_empty() {
                self.detected_levels = Some(result);
                return;
            }
        }

        // Fallback: detect levels directly from a single expiry smile
        // This bypasses cross-expiry confirmation when 0D/1D/4D expiries are not available.
        let n_strikes = self.strikes.len();
        let n_times = self.times.len();
        let ti = self.selected_expiry.min(n_times.saturating_sub(1));

        // Prefer local vol (if available) over IV for dislocation detection
        let local_vols: Vec<f64> = if let Some(ref lv_surf) = self.local_vol_surface {
            let t = self.times[ti];
            (0..n_strikes)
                .map(|si| lv_surf.local_vol(self.strikes[si], t))
                .collect()
        } else {
            (0..n_strikes).map(|si| self.iv_grid[[si, ti]]).collect()
        };

        let dislocation_config = DislocationConfig {
            threshold: self.level_threshold,
            window: 2,
            suppression_radius: 1,
            ..Default::default()
        };

        let expiry_bucket =
            ExpiryBucket::from_days(self.times[ti] * 365.0).unwrap_or(ExpiryBucket::OneD);
        let candidates = slv_options::levels::detect_dislocations(
            &self.strikes,
            &local_vols,
            expiry_bucket, // Placeholder for single-expiry detection
            &dislocation_config,
        );

        // Convert candidates directly to levels (skip cross-expiry confirmation)
        let strike_spacing = slv_options::levels::compute_strike_spacing(&self.strikes);
        let high_cut = self.level_threshold + 1.0;
        let medium_cut = self.level_threshold;
        let mut levels: Vec<Level> = candidates
            .into_iter()
            .map(|c| {
                let distance_strikes = (c.strike - self.spot) / strike_spacing;
                let priority = distance_strikes.abs() <= self.level_proximity_band;
                let confidence = if c.z_score.abs() >= high_cut {
                    Confidence::High
                } else if c.z_score.abs() >= medium_cut {
                    Confidence::Medium
                } else {
                    Confidence::Low
                };

                let mut z_by_expiry = ZScoreByExpiry::default();
                z_by_expiry.set(expiry_bucket, c.z_score);

                Level {
                    strike: c.strike,
                    kind: c.kind,
                    confidence,
                    score: c.z_score.abs(),
                    confirm_expiries: vec![expiry_bucket],
                    z_by_expiry,
                    distance_strikes,
                    priority,
                    render_style: RenderStyle::from_level(confidence, priority),
                }
            })
            .collect();

        // Sort by distance from spot
        levels.sort_by(|a, b| {
            a.distance_strikes
                .abs()
                .partial_cmp(&b.distance_strikes.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build result
        let mut candidates_per_expiry = std::collections::HashMap::new();
        candidates_per_expiry.insert("selected".to_string(), levels.len());

        self.detected_levels = Some(LevelDetectionResult {
            levels,
            spot: self.spot,
            strike_spacing,
            config,
            candidates_per_expiry,
        });
    }
}

impl eframe::App for SLVApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Side panel for controls
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("SLV Options");
            ui.separator();

            // Ticker input section
            ui.heading("Fetch Data");
            ui.horizontal(|ui| {
                ui.label("Ticker:");
                ui.text_edit_singleline(&mut self.ticker_input);
            });
            ui.horizontal(|ui| {
                ui.label("Expiries:");
                ui.add(egui::DragValue::new(&mut self.num_expiries).clamp_range(1..=12));
            });
            ui.horizontal(|ui| {
                if ui.button("Fetch").clicked() && !self.is_fetching {
                    self.fetch_options();
                }
                if ui.button("Load").clicked() {
                    self.load_data();
                }
            });
            if !self.fetch_status.is_empty() {
                ui.label(&self.fetch_status);
            }

            ui.separator();
            ui.label(format!("Symbol: {}", self.symbol));
            ui.label(format!("Spot: ${:.2}", self.spot));
            ui.label(format!("Strikes: {}", self.strikes.len()));
            ui.label(format!("Expiries: {}", self.times.len()));

            ui.separator();
            ui.heading("Display");
            ui.checkbox(&mut self.show_calls, "Show Implied Vol");
            ui.checkbox(&mut self.show_local_vol, "Show Local Vol");
            ui.checkbox(&mut self.show_levels, "Show Levels");
            ui.checkbox(&mut self.show_level_labels, "Show Level Labels");

            // Level detection settings
            ui.separator();
            ui.heading("Level Detection");
            let mut threshold_changed = false;
            let mut band_changed = false;

            ui.horizontal(|ui| {
                ui.label("Threshold:");
                if ui
                    .add(
                        egui::DragValue::new(&mut self.level_threshold)
                            .speed(0.1)
                            .clamp_range(1.0..=5.0),
                    )
                    .changed()
                {
                    threshold_changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Proximity:");
                if ui
                    .add(
                        egui::DragValue::new(&mut self.level_proximity_band)
                            .speed(0.5)
                            .clamp_range(2.0..=20.0),
                    )
                    .changed()
                {
                    band_changed = true;
                }
            });

            if threshold_changed || band_changed {
                self.detect_levels();
            }

            // Show detected levels summary
            if let Some(ref result) = self.detected_levels {
                ui.separator();
                ui.label(format!("Levels: {}", result.levels.len()));
                ui.label(format!("  Spikes: {}", result.spikes().len()));
                ui.label(format!("  Air Pockets: {}", result.air_pockets().len()));
                ui.label(format!("  Priority: {}", result.priority_levels().len()));
            }

            let mut expiry_changed = false;
            if !self.times.is_empty() {
                ui.separator();
                ui.heading("Expiry");
                let times_clone: Vec<f64> = self.times.clone();
                for (i, t) in times_clone.iter().enumerate() {
                    let days = (t * 365.0) as i32;
                    if ui
                        .selectable_label(self.selected_expiry == i, format!("{}d", days))
                        .clicked()
                    {
                        self.selected_expiry = i;
                        expiry_changed = true;
                    }
                }
            }
            if expiry_changed {
                self.detect_levels();
            }

            ui.separator();
            ui.heading("Heston Params");
            ui.add(egui::Slider::new(&mut self.heston_v0, 0.01..=0.25).text("v0"));
            ui.add(egui::Slider::new(&mut self.heston_kappa, 0.1..=10.0).text("kappa"));
            ui.add(egui::Slider::new(&mut self.heston_theta, 0.01..=0.25).text("theta"));
            ui.add(egui::Slider::new(&mut self.heston_sigma, 0.1..=1.0).text("sigma"));
            ui.add(egui::Slider::new(&mut self.heston_rho, -0.99..=0.0).text("rho"));

            ui.separator();
            ui.heading("SLV");
            ui.add(egui::Slider::new(&mut self.slv_alpha, 0.0..=1.0).text("alpha"));
        });

        // Main panel with plots
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Volatility Smile");

            if self.strikes.is_empty() {
                ui.label("Click 'Fetch' or 'Load' to load options data");
                return;
            }

            let ti = self.selected_expiry.min(self.times.len().saturating_sub(1));
            let days = (self.times.get(ti).unwrap_or(&0.0) * 365.0) as i32;
            ui.label(format!("Expiry: {} days | Spot: ${:.2}", days, self.spot));

            // Build plot data
            let mut iv_points: Vec<[f64; 2]> = Vec::new();
            let mut lv_points: Vec<[f64; 2]> = Vec::new();

            for (si, &strike) in self.strikes.iter().enumerate() {
                let iv = self.iv_grid[[si, ti]];
                if iv > 0.0 {
                    iv_points.push([strike, iv * 100.0]);
                }

                if self.show_local_vol {
                    if let Some(ref lv_surf) = self.local_vol_surface {
                        let t = self.times[ti];
                        let lv = lv_surf.local_vol(strike, t);
                        if lv > 0.01 && lv < 1.0 {
                            lv_points.push([strike, lv * 100.0]);
                        }
                    }
                }
            }

            Plot::new("vol_smile")
                .view_aspect(2.0)
                .x_axis_label("Strike")
                .y_axis_label("Volatility (%)")
                .legend(egui_plot::Legend::default())
                .show(ui, |plot_ui| {
                    // Draw levels first (behind the curves)
                    if self.show_levels {
                        if let Some(ref result) = self.detected_levels {
                            for level in &result.levels {
                                let color = match level.kind {
                                    LevelKind::Spike => {
                                        // Red/orange for walls/pivots
                                        match level.confidence {
                                            Confidence::High => {
                                                egui::Color32::from_rgba_unmultiplied(
                                                    255, 100, 100, 200,
                                                )
                                            }
                                            Confidence::Medium => {
                                                egui::Color32::from_rgba_unmultiplied(
                                                    255, 150, 100, 150,
                                                )
                                            }
                                            Confidence::Low => {
                                                egui::Color32::from_rgba_unmultiplied(
                                                    255, 180, 150, 100,
                                                )
                                            }
                                        }
                                    }
                                    LevelKind::AirPocket => {
                                        // Cyan/blue for acceleration zones
                                        match level.confidence {
                                            Confidence::High => {
                                                egui::Color32::from_rgba_unmultiplied(
                                                    100, 200, 255, 200,
                                                )
                                            }
                                            Confidence::Medium => {
                                                egui::Color32::from_rgba_unmultiplied(
                                                    100, 180, 255, 150,
                                                )
                                            }
                                            Confidence::Low => {
                                                egui::Color32::from_rgba_unmultiplied(
                                                    150, 200, 255, 100,
                                                )
                                            }
                                        }
                                    }
                                };

                                let width = match level.confidence {
                                    Confidence::High => 2.0,
                                    Confidence::Medium => 1.5,
                                    Confidence::Low => 1.0,
                                };

                                let name = if self.show_level_labels {
                                    format!(
                                        "{:.0} {}",
                                        level.strike,
                                        if level.kind == LevelKind::Spike {
                                            "W"
                                        } else {
                                            "A"
                                        }
                                    )
                                } else {
                                    String::new()
                                };

                                plot_ui.vline(
                                    VLine::new(level.strike)
                                        .name(name)
                                        .color(color)
                                        .width(width)
                                        .style(if level.priority {
                                            egui_plot::LineStyle::Solid
                                        } else {
                                            egui_plot::LineStyle::Dashed { length: 4.0 }
                                        }),
                                );
                            }
                        }
                    }

                    // Implied vol curve
                    if self.show_calls && !iv_points.is_empty() {
                        plot_ui.line(
                            Line::new(PlotPoints::new(iv_points.clone()))
                                .name("Implied Vol")
                                .color(egui::Color32::LIGHT_BLUE)
                                .width(2.0),
                        );
                    }

                    // Local vol curve (green)
                    if self.show_local_vol && !lv_points.is_empty() {
                        plot_ui.line(
                            Line::new(PlotPoints::new(lv_points))
                                .name("Local Vol")
                                .color(egui::Color32::LIGHT_GREEN)
                                .width(2.0),
                        );
                    }

                    // ATM line (yellow)
                    plot_ui.vline(
                        VLine::new(self.spot)
                            .name("ATM")
                            .color(egui::Color32::YELLOW)
                            .width(1.5)
                            .style(egui_plot::LineStyle::Dashed { length: 5.0 }),
                    );
                });

            // Levels table
            if self.show_levels {
                if let Some(ref result) = self.detected_levels {
                    if !result.levels.is_empty() {
                        ui.separator();
                        ui.heading("Detected Levels");

                        egui::ScrollArea::vertical()
                            .max_height(200.0)
                            .show(ui, |ui| {
                                egui::Grid::new("levels_grid")
                                    .striped(true)
                                    .spacing([20.0, 4.0])
                                    .show(ui, |ui| {
                                        // Header
                                        ui.strong("Strike");
                                        ui.strong("Type");
                                        ui.strong("Confidence");
                                        ui.strong("Distance");
                                        ui.strong("Expiries");
                                        ui.end_row();

                                        for level in &result.levels {
                                            ui.label(format!("{:.0}", level.strike));

                                            let type_label = match level.kind {
                                                LevelKind::Spike => "Wall/Pivot",
                                                LevelKind::AirPocket => "Acceleration",
                                            };
                                            ui.label(type_label);

                                            let conf_label = match level.confidence {
                                                Confidence::High => "High",
                                                Confidence::Medium => "Medium",
                                                Confidence::Low => "Low",
                                            };
                                            ui.label(conf_label);

                                            ui.label(format!("{:+.1}", level.distance_strikes));

                                            let expiries: Vec<&str> = level
                                                .confirm_expiries
                                                .iter()
                                                .map(|e| e.label())
                                                .collect();
                                            ui.label(expiries.join(", "));

                                            ui.end_row();
                                        }
                                    });
                            });
                    }
                }
            }

            ui.separator();

            // Term structure plot
            ui.heading("Term Structure (ATM)");

            let atm_idx = self
                .strikes
                .iter()
                .position(|&k| k >= self.spot)
                .unwrap_or(self.strikes.len() / 2);

            let mut term_points: Vec<[f64; 2]> = Vec::new();
            for (ti, &t) in self.times.iter().enumerate() {
                let iv = self.iv_grid[[atm_idx, ti]];
                if iv > 0.0 {
                    term_points.push([t * 365.0, iv * 100.0]);
                }
            }

            Plot::new("term_structure")
                .view_aspect(3.0)
                .x_axis_label("Days to Expiry")
                .y_axis_label("ATM Vol (%)")
                .show(ui, |plot_ui| {
                    if !term_points.is_empty() {
                        plot_ui.line(
                            Line::new(PlotPoints::new(term_points))
                                .name("ATM IV")
                                .color(egui::Color32::LIGHT_BLUE)
                                .width(2.0),
                        );
                    }
                });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 900.0])
            .with_title("SLV Options - Volatility Surface & Levels"),
        ..Default::default()
    };

    eframe::run_native(
        "SLV Options",
        options,
        Box::new(|_cc| Box::new(SLVApp::default())),
    )
}
