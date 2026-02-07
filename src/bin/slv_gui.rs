//! SLV Options GUI
//!
//! Interactive visualization of volatility surfaces, SLV model, and detected levels.

use chrono::{NaiveDate, Utc};
use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoints, VLine};
use ndarray::Array2;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::sync::mpsc;
use std::time::{Duration, Instant};

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

enum FetchCommand {
    Fetch { symbol: String, num_expiries: usize },
    Shutdown,
}

enum FetchResult {
    SurfaceReady {
        symbol: String,
        data: OptionsData,
        candles: Vec<CandleBar>,
        fetch_duration: Duration,
    },
    Error {
        symbol: String,
        message: String,
        fetch_duration: Duration,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MainTab {
    PriceChart,
    VolatilitySmile,
    Levels,
    TermStructure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DrawMode {
    None,
    Horizontal,
    TrendLine,
}

fn to_i64(value: Option<u64>) -> i64 {
    match value {
        Some(v) if v > i64::MAX as u64 => i64::MAX,
        Some(v) => v as i64,
        None => 0,
    }
}

fn option_quote_to_option_data(quote: &OptionQuote) -> OptionData {
    OptionData {
        strike: quote.contract.strike,
        bid: quote.bid,
        ask: quote.ask,
        last: quote.last,
        iv: quote.implied_vol,
        volume: to_i64(quote.volume),
        open_interest: to_i64(quote.open_interest),
    }
}

fn quote_surface_to_options_data(surface: &QuoteSurface) -> OptionsData {
    let chains = surface
        .chains
        .iter()
        .map(|chain| ChainData {
            expiry: chain.expiry.format("%Y-%m-%d").to_string(),
            calls: chain
                .calls
                .iter()
                .map(option_quote_to_option_data)
                .collect(),
            puts: chain.puts.iter().map(option_quote_to_option_data).collect(),
        })
        .collect();

    OptionsData {
        symbol: surface.underlying.clone(),
        spot: surface.spot,
        timestamp: surface.timestamp.to_rfc3339(),
        chains,
    }
}

struct SLVApp {
    cmd_tx: mpsc::Sender<FetchCommand>,
    result_rx: mpsc::Receiver<FetchResult>,

    symbol: String,
    spot: f64,
    strikes: Vec<f64>,
    times: Vec<f64>,
    iv_grid: Array2<f64>,

    main_tab: MainTab,
    selected_expiry: usize,
    show_calls: bool,
    show_local_vol: bool,
    show_levels: bool,
    show_level_labels: bool,
    show_settings_window: bool,
    draw_mode: DrawMode,
    chart_visible_bars: usize,
    drawn_hlines: Vec<f64>,
    drawn_trendlines: Vec<([f64; 2], [f64; 2])>,
    pending_trend_start: Option<[f64; 2]>,

    ticker_input: String,
    num_expiries: u32,
    fetch_status: String,
    is_fetching: bool,
    auto_refresh: bool,
    refresh_interval_secs: u32,
    last_refresh: Option<Instant>,
    next_refresh_at: Option<Instant>,
    last_fetch_duration: Option<Duration>,
    consecutive_errors: u32,
    last_error: Option<String>,

    level_threshold: f64,
    level_proximity_band: f64,

    heston_v0: f64,
    heston_kappa: f64,
    heston_theta: f64,
    heston_sigma: f64,
    heston_rho: f64,
    slv_alpha: f64,

    local_vol_surface: Option<LocalVolSurface>,
    vol_surface: Option<VolSurface>,
    detected_levels: Option<LevelDetectionResult>,
    candles: Vec<CandleBar>,
}

impl SLVApp {
    fn apply_visual_style(ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(8.0, 5.0);
        style.spacing.interact_size.y = 24.0;
        style.spacing.slider_width = 150.0;
        style
            .text_styles
            .insert(egui::TextStyle::Heading, egui::FontId::proportional(18.0));
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(13.0));
        style
            .text_styles
            .insert(egui::TextStyle::Button, egui::FontId::proportional(13.0));

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(14, 16, 20);
        visuals.window_fill = egui::Color32::from_rgb(16, 19, 24);
        visuals.extreme_bg_color = egui::Color32::from_rgb(20, 24, 30);
        visuals.faint_bg_color = egui::Color32::from_rgb(24, 29, 36);
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(18, 22, 27);
        visuals.widgets.noninteractive.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(36, 41, 50));
        visuals.widgets.noninteractive.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(178, 186, 197));
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(24, 29, 36);
        visuals.widgets.inactive.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(44, 50, 61));
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(30, 38, 49);
        visuals.widgets.hovered.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(74, 92, 116));
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(34, 47, 64);
        visuals.widgets.active.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(92, 118, 150));
        visuals.selection.bg_fill = egui::Color32::from_rgb(10, 132, 255);
        visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
        visuals.hyperlink_color = egui::Color32::from_rgb(90, 170, 255);
        visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(36, 42, 52));

        style.visuals = visuals;
        ctx.set_style(style);
    }

    fn sort_levels_by_price(levels: &mut [Level]) {
        levels.sort_by(|a, b| {
            a.strike
                .partial_cmp(&b.strike)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn moving_average_points(candles: &[CandleBar], period: usize) -> Vec<[f64; 2]> {
        if period == 0 || candles.len() < period {
            return Vec::new();
        }

        let mut points = Vec::with_capacity(candles.len() - period + 1);
        let mut rolling_sum = 0.0;

        for (i, candle) in candles.iter().enumerate() {
            rolling_sum += candle.close;
            if i >= period {
                rolling_sum -= candles[i - period].close;
            }
            if i + 1 >= period {
                points.push([i as f64, rolling_sum / period as f64]);
            }
        }

        points
    }

    fn new(cmd_tx: mpsc::Sender<FetchCommand>, result_rx: mpsc::Receiver<FetchResult>) -> Self {
        Self {
            cmd_tx,
            result_rx,
            symbol: "SPY".to_string(),
            spot: 500.0,
            strikes: vec![],
            times: vec![],
            iv_grid: Array2::zeros((0, 0)),
            main_tab: MainTab::PriceChart,
            selected_expiry: 0,
            show_calls: true,
            show_local_vol: true,
            show_levels: true,
            show_level_labels: true,
            show_settings_window: false,
            draw_mode: DrawMode::None,
            chart_visible_bars: 240,
            drawn_hlines: Vec::new(),
            drawn_trendlines: Vec::new(),
            pending_trend_start: None,
            ticker_input: "SPY".to_string(),
            num_expiries: 6,
            fetch_status: String::new(),
            is_fetching: false,
            auto_refresh: false,
            refresh_interval_secs: 60,
            last_refresh: None,
            next_refresh_at: None,
            last_fetch_duration: None,
            consecutive_errors: 0,
            last_error: None,
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
            candles: Vec::new(),
        }
    }

    fn normalized_ticker(&self) -> Option<String> {
        let ticker = self.ticker_input.trim().to_uppercase();
        if ticker.is_empty() {
            None
        } else {
            Some(ticker)
        }
    }

    fn schedule_next_regular_refresh(&mut self) {
        if self.auto_refresh {
            self.next_refresh_at =
                Some(Instant::now() + Duration::from_secs(self.refresh_interval_secs as u64));
        }
    }

    fn schedule_backoff_refresh(&mut self) {
        if self.auto_refresh {
            let exp = self.consecutive_errors.saturating_sub(1).min(6);
            let backoff_secs = (5_u64 * (1_u64 << exp)).min(300);
            self.next_refresh_at = Some(Instant::now() + Duration::from_secs(backoff_secs));
        }
    }

    fn request_fetch(&mut self) -> bool {
        if self.is_fetching {
            return false;
        }

        let Some(symbol) = self.normalized_ticker() else {
            self.fetch_status = "Enter a ticker symbol".to_string();
            return false;
        };

        let send_result = self.cmd_tx.send(FetchCommand::Fetch {
            symbol: symbol.clone(),
            num_expiries: self.num_expiries as usize,
        });

        match send_result {
            Ok(()) => {
                self.fetch_status = format!("Fetching {}...", symbol);
                self.is_fetching = true;
                self.last_error = None;
                true
            }
            Err(_) => {
                self.fetch_status = "Fetcher thread is unavailable".to_string();
                self.is_fetching = false;
                false
            }
        }
    }

    fn load_data_from_disk(&mut self) {
        let ticker = self.ticker_input.trim().to_lowercase();
        if ticker.is_empty() {
            self.fetch_status = "Enter a ticker symbol".to_string();
            return;
        }

        let data_path = format!(
            "D:/Stochastic local volatility/slv-options/data/{}_options.json",
            ticker
        );

        let file = match File::open(&data_path) {
            Ok(f) => f,
            Err(e) => {
                self.fetch_status = format!("Could not open {}: {}", data_path, e);
                return;
            }
        };

        let reader = BufReader::new(file);
        let data: OptionsData = match serde_json::from_reader(reader) {
            Ok(d) => d,
            Err(e) => {
                self.fetch_status = format!("Failed to parse JSON: {}", e);
                return;
            }
        };

        match self.apply_options_data(data) {
            Ok(()) => {
                self.candles.clear();
                self.fetch_status = format!("Loaded {} from disk", self.symbol);
                self.last_error = None;
                self.consecutive_errors = 0;
            }
            Err(e) => {
                self.fetch_status = format!("Load failed: {}", e);
                self.last_error = Some(e);
            }
        }
    }

    fn handle_fetch_result(&mut self, result: FetchResult) {
        match result {
            FetchResult::SurfaceReady {
                symbol,
                data,
                candles,
                fetch_duration,
            } => {
                self.is_fetching = false;
                self.last_fetch_duration = Some(fetch_duration);

                let current_ticker = self.normalized_ticker().unwrap_or_default();
                if current_ticker != symbol {
                    self.fetch_status = format!(
                        "Ignored stale {} result (current ticker: {})",
                        symbol,
                        if current_ticker.is_empty() {
                            "<empty>"
                        } else {
                            &current_ticker
                        }
                    );
                    if self.auto_refresh {
                        self.next_refresh_at = Some(Instant::now());
                    }
                    return;
                }

                match self.apply_options_data(data) {
                    Ok(()) => {
                        self.candles = candles;
                        self.fetch_status =
                            format!("{} updated in {} ms", symbol, fetch_duration.as_millis());
                        self.last_refresh = Some(Instant::now());
                        self.last_error = None;
                        self.consecutive_errors = 0;
                        self.schedule_next_regular_refresh();
                    }
                    Err(e) => {
                        self.fetch_status = format!("{} update failed: {}", symbol, e);
                        self.last_error = Some(e);
                        self.consecutive_errors = self.consecutive_errors.saturating_add(1);
                        self.schedule_backoff_refresh();
                    }
                }
            }
            FetchResult::Error {
                symbol,
                message,
                fetch_duration,
            } => {
                self.is_fetching = false;
                self.last_fetch_duration = Some(fetch_duration);
                self.last_error = Some(message.clone());
                self.consecutive_errors = self.consecutive_errors.saturating_add(1);
                self.fetch_status = format!(
                    "{} fetch error ({} ms): {}",
                    symbol,
                    fetch_duration.as_millis(),
                    message
                );
                self.schedule_backoff_refresh();
            }
        }
    }

    fn poll_fetch_results(&mut self) {
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => self.handle_fetch_result(result),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.is_fetching = false;
                    self.fetch_status = "Fetcher thread disconnected".to_string();
                    break;
                }
            }
        }
    }

    fn apply_options_data(&mut self, data: OptionsData) -> Result<(), String> {
        self.symbol = data.symbol.clone();
        self.spot = data.spot;
        self.ticker_input = self.symbol.clone();

        let today = Utc::now().date_naive();
        let rate = 0.045;
        let div = 0.013;

        let valid_chains: Vec<(NaiveDate, &ChainData, f64)> = data
            .chains
            .iter()
            .filter_map(|chain| {
                let expiry = NaiveDate::parse_from_str(&chain.expiry, "%Y-%m-%d").ok()?;
                let tte = (expiry - today).num_days() as f64 / 365.25;
                if tte > 0.0 {
                    Some((expiry, chain, tte))
                } else {
                    None
                }
            })
            .collect();

        if valid_chains.is_empty() {
            return Err("No valid future expiries found".to_string());
        }

        let mut strikes: Vec<f64> = Vec::new();
        for (_, chain, _) in &valid_chains {
            for call in &chain.calls {
                if call.strike.is_finite()
                    && call.strike > 0.0
                    && !strikes.iter().any(|k| (k - call.strike).abs() < 0.001)
                {
                    strikes.push(call.strike);
                }
            }
        }

        if strikes.is_empty() {
            return Err("No valid call strikes found".to_string());
        }

        strikes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let atm_idx = strikes
            .iter()
            .position(|&k| k >= self.spot)
            .unwrap_or(strikes.len() / 2);
        let start = atm_idx.saturating_sub(20);
        let end = (atm_idx + 20).min(strikes.len());
        let filtered_strikes = strikes[start..end].to_vec();

        if filtered_strikes.is_empty() {
            return Err("No strikes left after ATM filtering".to_string());
        }

        let mut raw_iv_grid: Array2<f64> =
            Array2::zeros((filtered_strikes.len(), valid_chains.len()));

        for (ti, (_, chain, _)) in valid_chains.iter().enumerate() {
            for call in &chain.calls {
                if let Some(si) = filtered_strikes
                    .iter()
                    .position(|&k| (k - call.strike).abs() < 0.01)
                {
                    if let Some(iv) = call.iv {
                        if iv.is_finite() && iv > 0.01 && iv < 3.0 {
                            raw_iv_grid[[si, ti]] = iv;
                        }
                    }
                }
            }

            for put in &chain.puts {
                if let Some(si) = filtered_strikes
                    .iter()
                    .position(|&k| (k - put.strike).abs() < 0.01)
                {
                    if raw_iv_grid[[si, ti]] == 0.0 {
                        if let Some(iv) = put.iv {
                            if iv.is_finite() && iv > 0.01 && iv < 3.0 {
                                raw_iv_grid[[si, ti]] = iv;
                            }
                        }
                    }
                }
            }
        }

        let valid_cols: Vec<usize> = (0..valid_chains.len())
            .filter(|&ti| (0..filtered_strikes.len()).any(|si| raw_iv_grid[[si, ti]] > 0.0))
            .collect();

        if valid_cols.is_empty() {
            return Err("No implied vol data found in selected expiries".to_string());
        }

        let mut times: Vec<f64> = Vec::with_capacity(valid_cols.len());
        let mut expiries: Vec<NaiveDate> = Vec::with_capacity(valid_cols.len());
        let mut iv_grid: Array2<f64> = Array2::zeros((filtered_strikes.len(), valid_cols.len()));

        for (new_ti, old_ti) in valid_cols.iter().enumerate() {
            times.push(valid_chains[*old_ti].2);
            expiries.push(valid_chains[*old_ti].0);
            for si in 0..filtered_strikes.len() {
                iv_grid[[si, new_ti]] = raw_iv_grid[[si, *old_ti]];
            }
        }

        for ti in 0..times.len() {
            let mut last_valid = None;
            for si in 0..filtered_strikes.len() {
                let v = iv_grid[[si, ti]];
                if v > 0.0 {
                    last_valid = Some(v);
                } else if let Some(prev) = last_valid {
                    iv_grid[[si, ti]] = prev;
                }
            }

            let mut last_valid = None;
            for si in (0..filtered_strikes.len()).rev() {
                let v = iv_grid[[si, ti]];
                if v > 0.0 {
                    last_valid = Some(v);
                } else if let Some(next) = last_valid {
                    iv_grid[[si, ti]] = next;
                }
            }
        }

        self.strikes = filtered_strikes.clone();
        self.times = times.clone();
        self.iv_grid = iv_grid.clone();
        self.selected_expiry = self.selected_expiry.min(self.times.len().saturating_sub(1));

        let vol_surface = VolSurface::from_grid(
            &self.symbol,
            self.spot,
            today,
            filtered_strikes,
            times,
            expiries,
            iv_grid,
            rate,
            div,
        );

        self.local_vol_surface = LocalVolSurface::from_implied_vol(&vol_surface).ok();
        self.vol_surface = Some(vol_surface);
        self.detect_levels();

        Ok(())
    }

    fn detect_levels(&mut self) {
        if self.strikes.is_empty() || self.times.is_empty() {
            self.detected_levels = None;
            return;
        }

        let mut config = LevelConfig::default();
        config.dislocation.threshold = self.level_threshold;
        config.proximity.priority_band = self.level_proximity_band;

        if let Some(ref lv_surface) = self.local_vol_surface {
            let detector = LevelDetector::with_config(config.clone());
            let mut result = detector.detect(lv_surface, self.spot);
            if !result.levels.is_empty() {
                Self::sort_levels_by_price(&mut result.levels);
                self.detected_levels = Some(result);
                return;
            }
        }

        let n_strikes = self.strikes.len();
        let n_times = self.times.len();
        let ti = self.selected_expiry.min(n_times.saturating_sub(1));

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
            expiry_bucket,
            &dislocation_config,
        );

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

        Self::sort_levels_by_price(&mut levels);

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

    fn render_price_chart_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading(format!("{} Price Chart", self.symbol));

        if self.candles.is_empty() {
            ui.label("No market candles loaded yet. Click 'Fetch Now'.");
            return;
        }

        self.chart_visible_bars = self.chart_visible_bars.clamp(40, 1200);
        let visible_bars = self.candles.len().min(self.chart_visible_bars);
        let visible = &self.candles[self.candles.len() - visible_bars..];

        let last = visible.last().expect("checked non-empty candles");
        let prev_close = visible
            .iter()
            .rev()
            .nth(1)
            .map(|c| c.close)
            .unwrap_or(last.open);
        let delta = last.close - prev_close;
        let pct = if prev_close.abs() > f64::EPSILON {
            (delta / prev_close) * 100.0
        } else {
            0.0
        };
        let delta_color = if delta >= 0.0 {
            egui::Color32::from_rgb(40, 190, 120)
        } else {
            egui::Color32::from_rgb(220, 80, 80)
        };

        let last_volume = last.volume.unwrap_or(0);
        let avg_volume = {
            let vols: Vec<u64> = visible.iter().filter_map(|c| c.volume).collect();
            if vols.is_empty() {
                0
            } else {
                (vols.iter().sum::<u64>() / vols.len() as u64) as u64
            }
        };

        ui.label(format!(
            "Last bar: {}",
            last.timestamp.format("%Y-%m-%d %H:%M UTC"),
        ));
        ui.horizontal(|ui| {
            ui.label("Bars:");
            ui.add(egui::Slider::new(&mut self.chart_visible_bars, 40..=1200));
            ui.separator();
            ui.label("Draw:");
            ui.selectable_value(&mut self.draw_mode, DrawMode::None, "None");
            ui.selectable_value(&mut self.draw_mode, DrawMode::Horizontal, "H-Line");
            ui.selectable_value(&mut self.draw_mode, DrawMode::TrendLine, "Trend");
            if ui.button("Clear Drawings").clicked() {
                self.drawn_hlines.clear();
                self.drawn_trendlines.clear();
                self.pending_trend_start = None;
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("Close {:.2}", last.close));
            ui.colored_label(delta_color, format!("{:+.2} ({:+.2}%)", delta, pct));
            ui.separator();
            ui.label(format!("O {:.2}", last.open));
            ui.label(format!("H {:.2}", last.high));
            ui.label(format!("L {:.2}", last.low));
            ui.separator();
            ui.label(format!("Bars {} / {}", visible_bars, self.candles.len()));
            ui.separator();
            ui.label(format!("Vol {:>8}", last_volume));
            ui.label(format!("AvgVol {:>8}", avg_volume));
        });
        ui.label("Navigation: drag to pan, wheel to zoom, right-drag for boxed zoom, double-click to reset. Drawing: click on chart.");

        let x_start = 0.0;
        let x_end = (visible_bars.saturating_sub(1) as f64).max(1.0);
        let sma20 = Self::moving_average_points(visible, 20);
        let sma50 = Self::moving_average_points(visible, 50);

        let mut y_min = visible.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
        let mut y_max = visible
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max);
        y_min = y_min.min(self.spot);
        y_max = y_max.max(self.spot);
        if self.show_levels {
            if let Some(ref result) = self.detected_levels {
                for level in &result.levels {
                    y_min = y_min.min(level.strike);
                    y_max = y_max.max(level.strike);
                }
            }
        }
        let pad = ((y_max - y_min) * 0.07).max(0.5);

        let mut click_point: Option<egui_plot::PlotPoint> = None;
        let mut click_on_plot = false;

        Plot::new("price_candles")
            .height(520.0)
            .view_aspect(2.2)
            .x_axis_label("Recent Bars")
            .y_axis_label("Price")
            .allow_drag(true)
            .allow_zoom(true)
            .allow_scroll(true)
            .allow_boxed_zoom(true)
            .include_x(x_start)
            .include_x(x_end)
            .include_y(y_min - pad)
            .include_y(y_max + pad)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                if plot_ui.response().hovered() {
                    click_point = plot_ui.pointer_coordinate();
                    click_on_plot = plot_ui.response().clicked_by(egui::PointerButton::Primary);
                }

                let body_width = if visible_bars > 180 { 2.4 } else { 3.4 };

                for (i, candle) in visible.iter().enumerate() {
                    let x = i as f64;

                    plot_ui.line(
                        Line::new(PlotPoints::new(vec![[x, candle.low], [x, candle.high]]))
                            .color(egui::Color32::from_rgb(140, 146, 155))
                            .width(1.0),
                    );

                    let body_color = if candle.close >= candle.open {
                        egui::Color32::from_rgb(40, 190, 120)
                    } else {
                        egui::Color32::from_rgb(220, 80, 80)
                    };
                    plot_ui.line(
                        Line::new(PlotPoints::new(vec![[x, candle.open], [x, candle.close]]))
                            .color(body_color)
                            .width(body_width),
                    );
                }

                if !sma20.is_empty() {
                    plot_ui.line(
                        Line::new(PlotPoints::new(sma20))
                            .name("SMA 20")
                            .color(egui::Color32::from_rgb(98, 176, 255))
                            .width(1.6),
                    );
                }
                if !sma50.is_empty() {
                    plot_ui.line(
                        Line::new(PlotPoints::new(sma50))
                            .name("SMA 50")
                            .color(egui::Color32::from_rgb(255, 186, 88))
                            .width(1.8),
                    );
                }

                for (idx, y) in self.drawn_hlines.iter().enumerate() {
                    plot_ui.hline(
                        HLine::new(*y)
                            .name(format!("Draw H{}", idx + 1))
                            .color(egui::Color32::from_rgb(214, 214, 214))
                            .width(1.0),
                    );
                }

                for (idx, (a, b)) in self.drawn_trendlines.iter().enumerate() {
                    plot_ui.line(
                        Line::new(PlotPoints::new(vec![*a, *b]))
                            .name(format!("Draw T{}", idx + 1))
                            .color(egui::Color32::from_rgb(214, 214, 214))
                            .width(1.2),
                    );
                }

                if self.draw_mode == DrawMode::TrendLine {
                    if let (Some(start), Some(cursor)) = (self.pending_trend_start, click_point) {
                        plot_ui.line(
                            Line::new(PlotPoints::new(vec![start, [cursor.x, cursor.y]]))
                                .name("Trend Preview")
                                .color(egui::Color32::from_rgb(250, 250, 250))
                                .width(1.0)
                                .style(egui_plot::LineStyle::Dashed { length: 5.0 }),
                        );
                    }
                }

                if self.show_levels {
                    if let Some(ref result) = self.detected_levels {
                        for level in &result.levels {
                            let color = match level.kind {
                                LevelKind::Spike => match level.confidence {
                                    Confidence::High => {
                                        egui::Color32::from_rgba_unmultiplied(255, 100, 100, 200)
                                    }
                                    Confidence::Medium => {
                                        egui::Color32::from_rgba_unmultiplied(255, 150, 100, 150)
                                    }
                                    Confidence::Low => {
                                        egui::Color32::from_rgba_unmultiplied(255, 180, 150, 100)
                                    }
                                },
                                LevelKind::AirPocket => match level.confidence {
                                    Confidence::High => {
                                        egui::Color32::from_rgba_unmultiplied(100, 200, 255, 200)
                                    }
                                    Confidence::Medium => {
                                        egui::Color32::from_rgba_unmultiplied(100, 180, 255, 150)
                                    }
                                    Confidence::Low => {
                                        egui::Color32::from_rgba_unmultiplied(150, 200, 255, 100)
                                    }
                                },
                            };

                            let width = match level.confidence {
                                Confidence::High => 2.0,
                                Confidence::Medium => 1.5,
                                Confidence::Low => 1.0,
                            };

                            let label = if self.show_level_labels {
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

                            plot_ui.line(
                                Line::new(PlotPoints::new(vec![
                                    [x_start, level.strike],
                                    [x_end, level.strike],
                                ]))
                                .name(label)
                                .color(color)
                                .width(width)
                                .style(if level.priority {
                                    egui_plot::LineStyle::Solid
                                } else {
                                    egui_plot::LineStyle::Dashed { length: 5.0 }
                                }),
                            );
                        }
                    }
                }

                plot_ui.line(
                    Line::new(PlotPoints::new(vec![[x_start, self.spot], [x_end, self.spot]]))
                        .name("Reference Spot")
                        .color(egui::Color32::YELLOW)
                        .width(1.5)
                        .style(egui_plot::LineStyle::Dashed { length: 6.0 }),
                );
            });

        if click_on_plot {
            if let Some(point) = click_point {
                match self.draw_mode {
                    DrawMode::None => {}
                    DrawMode::Horizontal => self.drawn_hlines.push(point.y),
                    DrawMode::TrendLine => {
                        if let Some(start) = self.pending_trend_start.take() {
                            self.drawn_trendlines.push((start, [point.x, point.y]));
                        } else {
                            self.pending_trend_start = Some([point.x, point.y]);
                        }
                    }
                }
            }
        }

        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.pending_trend_start = None;
            self.draw_mode = DrawMode::None;
        }
    }
}

impl Drop for SLVApp {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(FetchCommand::Shutdown);
    }
}

impl eframe::App for SLVApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        Self::apply_visual_style(ctx);
        self.poll_fetch_results();

        if self.auto_refresh && !self.is_fetching {
            let now = Instant::now();
            let should_fetch = match self.next_refresh_at {
                Some(next) => now >= next,
                None => true,
            };

            if should_fetch {
                if self.request_fetch() {
                    self.next_refresh_at = None;
                } else {
                    self.next_refresh_at = Some(now + Duration::from_secs(5));
                }
            }
        }

        if self.auto_refresh || self.is_fetching {
            ctx.request_repaint_after(Duration::from_secs(1));
        }

        egui::TopBottomPanel::top("app_header")
            .exact_height(38.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("SLV Options")
                            .strong()
                            .color(egui::Color32::from_rgb(214, 220, 229)),
                    );
                    ui.separator();
                    ui.label("Ticker:");
                    ui.add_sized([70.0, 22.0], egui::TextEdit::singleline(&mut self.ticker_input));
                    ui.label("Expiries:");
                    ui.add(egui::DragValue::new(&mut self.num_expiries).clamp_range(1..=12));

                    if ui
                        .add_enabled(!self.is_fetching, egui::Button::new("Fetch"))
                        .clicked()
                    {
                        let _ = self.request_fetch();
                    }

                    if ui.button("Settings").clicked() {
                        self.show_settings_window = true;
                    }

                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!("{}  Spot ${:.2}", self.symbol, self.spot))
                            .color(egui::Color32::from_rgb(182, 191, 204)),
                    );

                    if let Some(duration) = self.last_fetch_duration {
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("Latency {} ms", duration.as_millis()))
                                .color(egui::Color32::from_rgb(154, 169, 190)),
                        );
                    }

                    if self.is_fetching {
                        ui.separator();
                        ui.spinner();
                    }
                });
            });

        if self.show_settings_window {
            let mut open = self.show_settings_window;
            egui::Window::new("Settings")
                .open(&mut open)
                .default_size([480.0, 700.0])
                .vscroll(true)
                .show(ctx, |ui| {
                    ui.heading("Data");
                    ui.horizontal(|ui| {
                        if ui.button("Fetch Now").clicked() {
                            let _ = self.request_fetch();
                        }
                        if ui.button("Load File").clicked() {
                            self.load_data_from_disk();
                        }
                    });

                    let auto_refresh_changed =
                        ui.checkbox(&mut self.auto_refresh, "Auto Refresh").changed();
                    let interval_changed = ui
                        .add_enabled(
                            self.auto_refresh,
                            egui::Slider::new(&mut self.refresh_interval_secs, 15..=300)
                                .text("Interval (s)"),
                        )
                        .changed();

                    if auto_refresh_changed {
                        if self.auto_refresh {
                            self.next_refresh_at = Some(Instant::now());
                        } else {
                            self.next_refresh_at = None;
                        }
                    } else if interval_changed && self.auto_refresh {
                        self.schedule_next_regular_refresh();
                    }

                    if !self.fetch_status.is_empty() {
                        ui.label(&self.fetch_status);
                    }
                    if let Some(last) = self.last_refresh {
                        ui.label(format!("Last update: {}s ago", last.elapsed().as_secs()));
                    }
                    if let Some(duration) = self.last_fetch_duration {
                        ui.label(format!("Last fetch: {} ms", duration.as_millis()));
                    }
                    if self.auto_refresh {
                        if let Some(next) = self.next_refresh_at {
                            let secs = next.saturating_duration_since(Instant::now()).as_secs();
                            ui.label(format!("Next refresh: {}s", secs));
                        } else if self.is_fetching {
                            ui.label("Next refresh: waiting for fetch");
                        } else {
                            ui.label("Next refresh: due");
                        }
                    }
                    if let Some(error) = &self.last_error {
                        ui.colored_label(egui::Color32::LIGHT_RED, format!("Last error: {}", error));
                    }

                    ui.separator();
                    ui.heading("Display");
                    ui.checkbox(&mut self.show_calls, "Show Implied Vol");
                    ui.checkbox(&mut self.show_local_vol, "Show Local Vol");
                    ui.checkbox(&mut self.show_levels, "Show Levels");
                    ui.checkbox(&mut self.show_level_labels, "Show Level Labels");

                    ui.separator();
                    ui.heading("Chart Interaction");
                    ui.label("Use mouse drag/wheel/right-drag inside chart for pan/zoom.");
                    ui.horizontal(|ui| {
                        ui.label("Visible bars:");
                        ui.add(egui::Slider::new(&mut self.chart_visible_bars, 40..=1200));
                    });

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

                    if let Some(ref result) = self.detected_levels {
                        ui.separator();
                        ui.label(format!("Levels: {}", result.levels.len()));
                        ui.label(format!("  Spikes: {}", result.spikes().len()));
                        ui.label(format!("  Air Pockets: {}", result.air_pockets().len()));
                        ui.label(format!("  Priority: {}", result.priority_levels().len()));
                    }

                    if !self.times.is_empty() {
                        ui.separator();
                        ui.heading("Expiry");
                        let mut expiry_changed = false;
                        let times_clone: Vec<f64> = self.times.clone();
                        ui.horizontal_wrapped(|ui| {
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
                        });
                        if expiry_changed {
                            self.detect_levels();
                        }
                    }

                    ui.separator();
                    ui.heading("Model");
                    ui.label("Heston Params");
                    ui.add(egui::Slider::new(&mut self.heston_v0, 0.01..=0.25).text("v0"));
                    ui.add(egui::Slider::new(&mut self.heston_kappa, 0.1..=10.0).text("kappa"));
                    ui.add(egui::Slider::new(&mut self.heston_theta, 0.01..=0.25).text("theta"));
                    ui.add(egui::Slider::new(&mut self.heston_sigma, 0.1..=1.0).text("sigma"));
                    ui.add(egui::Slider::new(&mut self.heston_rho, -0.99..=0.0).text("rho"));

                    ui.separator();
                    ui.label("SLV");
                    ui.add(egui::Slider::new(&mut self.slv_alpha, 0.0..=1.0).text("alpha"));
                });
            self.show_settings_window = open;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.main_tab, MainTab::PriceChart, "Price Chart");
                ui.selectable_value(
                    &mut self.main_tab,
                    MainTab::VolatilitySmile,
                    "Volatility Smile",
                );
                ui.selectable_value(&mut self.main_tab, MainTab::Levels, "Detected Levels");
                ui.selectable_value(
                    &mut self.main_tab,
                    MainTab::TermStructure,
                    "Term Structure",
                );
            });
            ui.separator();

            match self.main_tab {
                MainTab::PriceChart => {
                    self.render_price_chart_tab(ui);
                }
                MainTab::VolatilitySmile => {
                    if self.strikes.is_empty() {
                        ui.label("Click 'Fetch' to load options data");
                        return;
                    }

                    let ti = self.selected_expiry.min(self.times.len().saturating_sub(1));
                    let days = (self.times.get(ti).unwrap_or(&0.0) * 365.0) as i32;
                    ui.label(format!("Expiry: {} days | Spot: ${:.2}", days, self.spot));

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
                        .height(520.0)
                        .view_aspect(2.0)
                        .x_axis_label("Strike")
                        .y_axis_label("Volatility (%)")
                        .legend(egui_plot::Legend::default())
                        .show(ui, |plot_ui| {
                            if self.show_levels {
                                if let Some(ref result) = self.detected_levels {
                                    for level in &result.levels {
                                        let color = match level.kind {
                                            LevelKind::Spike => match level.confidence {
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
                                            },
                                            LevelKind::AirPocket => match level.confidence {
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
                                            },
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

                            if self.show_calls && !iv_points.is_empty() {
                                plot_ui.line(
                                    Line::new(PlotPoints::new(iv_points.clone()))
                                        .name("Implied Vol")
                                        .color(egui::Color32::LIGHT_BLUE)
                                        .width(2.0),
                                );
                            }

                            if self.show_local_vol && !lv_points.is_empty() {
                                plot_ui.line(
                                    Line::new(PlotPoints::new(lv_points))
                                        .name("Local Vol")
                                        .color(egui::Color32::LIGHT_GREEN)
                                        .width(2.0),
                                );
                            }

                            plot_ui.vline(
                                VLine::new(self.spot)
                                    .name("ATM")
                                    .color(egui::Color32::YELLOW)
                                    .width(1.5)
                                    .style(egui_plot::LineStyle::Dashed { length: 5.0 }),
                            );
                        });
                }
                MainTab::Levels => {
                    ui.heading("Detected Levels");
                    if self.show_levels {
                        if let Some(ref result) = self.detected_levels {
                            if !result.levels.is_empty() {
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    egui::Grid::new("levels_grid")
                                        .striped(true)
                                        .spacing([20.0, 4.0])
                                        .show(ui, |ui| {
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
                            } else {
                                ui.label("No levels detected for this expiry.");
                            }
                        } else {
                            ui.label("No levels detected.");
                        }
                    } else {
                        ui.label("Level display is disabled.");
                    }
                }
                MainTab::TermStructure => {
                    ui.heading("Term Structure (ATM)");
                    if self.strikes.is_empty() || self.times.is_empty() {
                        ui.label("No options surface loaded yet.");
                        return;
                    }

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
                        .height(520.0)
                        .view_aspect(2.2)
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
                }
            }
        });
    }
}
fn spawn_fetcher_thread(
    ctx: egui::Context,
) -> (mpsc::Sender<FetchCommand>, mpsc::Receiver<FetchResult>) {
    let (cmd_tx, cmd_rx) = mpsc::channel::<FetchCommand>();
    let (result_tx, result_rx) = mpsc::channel::<FetchResult>();

    std::thread::spawn(move || {
        let client = YahooClient::new();

        while let Ok(command) = cmd_rx.recv() {
            match command {
                FetchCommand::Fetch {
                    symbol,
                    num_expiries,
                } => {
                    let started = Instant::now();
                    let response = match client.get_option_surface_limited(&symbol, num_expiries) {
                        Ok(surface) => {
                            let candles = match client.get_candles(&symbol, "5m", "5d") {
                                Ok(c) => c,
                                Err(e) => {
                                    tracing::warn!("Failed to fetch candles for {}: {}", symbol, e);
                                    Vec::new()
                                }
                            };

                            FetchResult::SurfaceReady {
                                symbol,
                                data: quote_surface_to_options_data(&surface),
                                candles,
                                fetch_duration: started.elapsed(),
                            }
                        }
                        Err(e) => FetchResult::Error {
                            symbol,
                            message: e.to_string(),
                            fetch_duration: started.elapsed(),
                        },
                    };

                    if result_tx.send(response).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
                FetchCommand::Shutdown => break,
            }
        }
    });

    (cmd_tx, result_rx)
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 860.0])
            .with_title("SLV Options - Volatility Surface & Levels"),
        ..Default::default()
    };

    eframe::run_native(
        "SLV Options",
        options,
        Box::new(|cc| {
            let (cmd_tx, result_rx) = spawn_fetcher_thread(cc.egui_ctx.clone());
            Box::new(SLVApp::new(cmd_tx, result_rx))
        }),
    )
}
