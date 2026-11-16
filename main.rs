use axum::{
    extract::Query,
    http::{header, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use tower_http::cors::CorsLayer;

/// Option Type: Call (CE) or Put (PE)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    #[serde(rename = "CE", alias = "ce", alias = "Call", alias = "call")]
    CE,
    #[serde(rename = "PE", alias = "pe", alias = "Put", alias = "put")]
    PE,
}

impl std::fmt::Display for OptionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionType::CE => write!(f, "CE"),
            OptionType::PE => write!(f, "PE"),
        }
    }
}

/// Query parameters for single option pricing calculation
#[derive(Debug, Clone, Deserialize)]
pub struct PricingQuery {
    pub option_type: String,
    pub spot: f64,
    pub strike: f64,
    pub volatility: f64,
    pub days_to_expiry: f64,
    pub risk_free_rate: Option<f64>,
    pub dividend_yield: Option<f64>,
    pub steps: Option<usize>,
    pub actual_premium: Option<f64>,
}

/// Calculation results returned for a single option
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PricingResult {
    pub spot_price: f64,
    pub strike_price: f64,
    pub option_type: String,
    pub days_to_expiry: f64,
    pub up_factor: f64,
    pub down_factor: f64,
    pub risk_neutral_probability: f64,
    pub theoretical_premium: f64,
    pub actual_premium: Option<f64>,
    pub absolute_difference: Option<f64>,
    pub percentage_difference: Option<f64>,
}

/// Simple error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Core CRR Binomial Option Pricing implementation in Rust
///
/// Formulas:
/// - T = Days / 365.0
/// - dt = T / steps
/// - u = exp(σ * sqrt(dt))
/// - d = 1 / u
/// - p = [exp((r - q) * dt) - d] / (u - d)
/// - Terminal payoffs: Call max(S_T - K, 0), Put max(K - S_T, 0)
/// - Backward induction to node (0, 0)
pub fn calculate_crr(
    spot: f64,
    strike: f64,
    volatility: f64,
    days_to_expiry: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    steps: usize,
    option_type: OptionType,
    actual_premium: Option<f64>,
) -> Result<PricingResult, String> {
    // 1. Validation
    if spot <= 0.0 {
        return Err("Spot price must be strictly greater than 0.".to_string());
    }
    if strike <= 0.0 {
        return Err("Strike price must be strictly greater than 0.".to_string());
    }
    if volatility <= 0.0 {
        return Err("Volatility must be strictly greater than 0.".to_string());
    }
    if days_to_expiry < 0.0 {
        return Err("Days to expiry cannot be negative.".to_string());
    }
    if steps == 0 {
        return Err("Number of steps must be at least 1.".to_string());
    }
    if steps > 1000 {
        return Err("Number of steps cannot exceed 1000.".to_string());
    }

    let t_years = days_to_expiry / 365.0;

    // Edge case: At expiration (0 days)
    if days_to_expiry == 0.0 || t_years <= 1e-12 {
        let intrinsic = match option_type {
            OptionType::CE => (spot - strike).max(0.0),
            OptionType::PE => (strike - spot).max(0.0),
        };
        let (abs_diff, pct_diff) = compute_diff(intrinsic, actual_premium);
        return Ok(PricingResult {
            spot_price: round2(spot),
            strike_price: round2(strike),
            option_type: option_type.to_string(),
            days_to_expiry,
            up_factor: 1.0,
            down_factor: 1.0,
            risk_neutral_probability: 0.5,
            theoretical_premium: round2(intrinsic),
            actual_premium: actual_premium.map(round2),
            absolute_difference: abs_diff.map(round2),
            percentage_difference: pct_diff.map(round2),
        });
    }

    // CRR Parameters
    let dt = t_years / (steps as f64);
    let u = (volatility * dt.sqrt()).exp();
    let d = 1.0 / u;

    let growth = ((risk_free_rate - dividend_yield) * dt).exp();
    let p = (growth - d) / (u - d);

    if p < 0.0 || p > 1.0 {
        return Err(format!(
            "Arbitrage condition violated: probability p = {:.4} is not in [0, 1]. Increase steps or check rate/volatility.",
            p
        ));
    }

    let discount = (-risk_free_rate * dt).exp();

    // Terminal payoffs at step N
    let mut values = vec![0.0; steps + 1];
    for j in 0..=steps {
        let spot_at_node = spot * u.powi(j as i32) * d.powi((steps - j) as i32);
        values[j] = match option_type {
            OptionType::CE => (spot_at_node - strike).max(0.0),
            OptionType::PE => (strike - spot_at_node).max(0.0),
        };
    }

    // Backward induction from step N-1 down to 0
    for i in (0..steps).rev() {
        for j in 0..=i {
            values[j] = discount * (p * values[j + 1] + (1.0 - p) * values[j]);
        }
    }

    let theoretical_premium = values[0].max(0.0);
    let (abs_diff, pct_diff) = compute_diff(theoretical_premium, actual_premium);

    Ok(PricingResult {
        spot_price: round2(spot),
        strike_price: round2(strike),
        option_type: option_type.to_string(),
        days_to_expiry,
        up_factor: round4(u),
        down_factor: round4(d),
        risk_neutral_probability: round4(p),
        theoretical_premium: round2(theoretical_premium),
        actual_premium: actual_premium.map(round2),
        absolute_difference: abs_diff.map(round2),
        percentage_difference: pct_diff.map(round2),
    })
}

fn compute_diff(theoretical: f64, actual: Option<f64>) -> (Option<f64>, Option<f64>) {
    match actual {
        Some(act) => {
            let abs_diff = (theoretical - act).abs();
            let pct_diff = if act > 0.0 {
                Some((abs_diff / act) * 100.0)
            } else {
                None
            };
            (Some(abs_diff), pct_diff)
        }
        None => (None, None),
    }
}

fn round2(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}

fn round4(val: f64) -> f64 {
    (val * 10_000.0).round() / 10_000.0
}

// ---------------------------------------------------------------------------
// Option Chain Data
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OptionChainItem {
    pub strike_price: f64,
    pub ce_premium: f64,
    pub pe_premium: f64,
    pub expiry_date: String,
    pub underlying_value: f64,
}

/// GET /api/option-chain
async fn option_chain_handler() -> Response {
    let strikes = [
        23800.0, 23900.0, 24000.0, 24100.0, 24200.0, 24300.0, 24400.0, 24500.0,
        24600.0, 24700.0, 24800.0, 24900.0, 25000.0, 25100.0, 25200.0,
    ];
    let ce_prems = [775.20, 682.50, 594.10, 508.30, 426.00, 348.50, 275.50, 209.80, 152.40, 104.60, 67.80, 41.20, 23.50, 12.80, 6.50];
    let pe_prems = [18.50, 25.40, 36.80, 52.20, 73.50, 102.00, 140.20, 192.50, 258.00, 338.50, 431.20, 532.80, 638.00, 744.50, 852.00];

    let mut items = Vec::new();
    for i in 0..strikes.len() {
        items.push(OptionChainItem {
            strike_price: strikes[i],
            ce_premium: ce_prems[i],
            pe_premium: pe_prems[i],
            expiry_date: "2024-11-28".to_string(),
            underlying_value: 24530.90,
        });
    }

    (StatusCode::OK, Json(items)).into_response()
}

// ---------------------------------------------------------------------------
// Dataset Processing (Upload & Calculation)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ProcessDatasetPayload {
    pub csv_content: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ProcessedRow {
    pub row_number: usize,
    pub date: String,
    pub option_type: String,
    pub spot_price: Option<f64>,
    pub strike_price: Option<f64>,
    pub expiry_date: String,
    pub actual_premium: Option<f64>,
    pub up_factor: Option<f64>,
    pub down_factor: Option<f64>,
    pub risk_neutral_probability: Option<f64>,
    pub theoretical_premium: Option<f64>,
    pub absolute_difference: Option<f64>,
    pub percentage_difference: Option<f64>,
    pub status: String,
    pub error_message: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ProcessDatasetResponse {
    pub total_records: usize,
    pub success_records: usize,
    pub error_records: usize,
    pub rows: Vec<ProcessedRow>,
}

/// POST /api/process-dataset
/// Reads every row in CSV, validates, calculates CRR parameters & theoretical premium, and compares
async fn process_dataset_handler(body: String) -> Response {
    // Support both raw CSV text and JSON payload {"csv_content": "..."}
    let csv_text = if let Ok(payload) = serde_json::from_str::<ProcessDatasetPayload>(&body) {
        payload.csv_content
    } else {
        body
    };

    let mut lines = csv_text.lines();
    let header_line = match lines.next() {
        Some(h) => h,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Uploaded CSV file is empty.".to_string(),
                }),
            )
                .into_response();
        }
    };

    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().to_lowercase().replace(' ', "_"))
        .collect();

    // Map column names to indices
    let idx_date = headers.iter().position(|h| h.contains("date") && !h.contains("expiry"));
    let idx_type = headers.iter().position(|h| h.contains("type"));
    let idx_spot = headers.iter().position(|h| h.contains("spot"));
    let idx_strike = headers.iter().position(|h| h.contains("strike"));
    let idx_expiry = headers.iter().position(|h| h.contains("expiry"));
    let idx_vol = headers.iter().position(|h| h.contains("vol"));
    let idx_rate = headers.iter().position(|h| h.contains("rate"));
    let idx_div = headers.iter().position(|h| h.contains("div") || h.contains("yield"));
    let idx_days = headers.iter().position(|h| h.contains("day"));
    let idx_steps = headers.iter().position(|h| h.contains("step"));
    let idx_actual = headers.iter().position(|h| h.contains("actual") || h.contains("premium"));

    if idx_type.is_none() || idx_spot.is_none() || idx_strike.is_none() || idx_vol.is_none() || idx_days.is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Missing required CSV columns. Required: Option_Type, Spot_Price, Strike_Price, Volatility, Days_To_Expiry.".to_string(),
            }),
        )
            .into_response();
    }

    let mut rows = Vec::new();
    let mut success_count = 0;
    let mut error_count = 0;
    let mut row_num = 1;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        row_num += 1;
        let cols: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();

        let date = idx_date.and_then(|i| cols.get(i)).unwrap_or(&"").to_string();
        let raw_type = idx_type.and_then(|i| cols.get(i)).unwrap_or(&"").to_string();
        let expiry = idx_expiry.and_then(|i| cols.get(i)).unwrap_or(&"").to_string();

        let opt_type = match raw_type.to_uppercase().as_str() {
            "CE" | "CALL" => Ok(OptionType::CE),
            "PE" | "PUT" => Ok(OptionType::PE),
            other => Err(format!("Invalid option type '{}'. Must be 'CE' or 'PE'.", other)),
        };

        let spot = idx_spot.and_then(|i| cols.get(i)).and_then(|v| v.parse::<f64>().ok());
        let strike = idx_strike.and_then(|i| cols.get(i)).and_then(|v| v.parse::<f64>().ok());
        let vol = idx_vol.and_then(|i| cols.get(i)).and_then(|v| v.parse::<f64>().ok());
        let days = idx_days.and_then(|i| cols.get(i)).and_then(|v| v.parse::<f64>().ok());
        let rate = idx_rate.and_then(|i| cols.get(i)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.065);
        let div = idx_div.and_then(|i| cols.get(i)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.012);
        let steps = idx_steps.and_then(|i| cols.get(i)).and_then(|v| v.parse::<usize>().ok()).unwrap_or(50);
        let actual = idx_actual.and_then(|i| cols.get(i)).and_then(|v| v.parse::<f64>().ok());

        // Validate row fields
        if let (Ok(o_type), Some(s_val), Some(k_val), Some(mut v_val), Some(d_val)) =
            (opt_type, spot, strike, vol, days)
        {
            if v_val > 1.0 {
                v_val /= 100.0;
            }
            let r_val = if rate > 1.0 { rate / 100.0 } else { rate };
            let q_val = if div > 1.0 { div / 100.0 } else { div };

            match calculate_crr(s_val, k_val, v_val, d_val, r_val, q_val, steps, o_type, actual) {
                Ok(calc) => {
                    success_count += 1;
                    rows.push(ProcessedRow {
                        row_number: row_num,
                        date,
                        option_type: o_type.to_string(),
                        spot_price: Some(s_val),
                        strike_price: Some(k_val),
                        expiry_date: expiry,
                        actual_premium: actual,
                        up_factor: Some(calc.up_factor),
                        down_factor: Some(calc.down_factor),
                        risk_neutral_probability: Some(calc.risk_neutral_probability),
                        theoretical_premium: Some(calc.theoretical_premium),
                        absolute_difference: calc.absolute_difference,
                        percentage_difference: calc.percentage_difference,
                        status: "SUCCESS".to_string(),
                        error_message: None,
                    });
                }
                Err(e) => {
                    error_count += 1;
                    rows.push(ProcessedRow {
                        row_number: row_num,
                        date,
                        option_type: raw_type,
                        spot_price: Some(s_val),
                        strike_price: Some(k_val),
                        expiry_date: expiry,
                        actual_premium: actual,
                        up_factor: None,
                        down_factor: None,
                        risk_neutral_probability: None,
                        theoretical_premium: None,
                        absolute_difference: None,
                        percentage_difference: None,
                        status: "ERROR".to_string(),
                        error_message: Some(e),
                    });
                }
            }
        } else {
            error_count += 1;
            rows.push(ProcessedRow {
                row_number: row_num,
                date,
                option_type: raw_type,
                spot_price: spot,
                strike_price: strike,
                expiry_date: expiry,
                actual_premium: actual,
                up_factor: None,
                down_factor: None,
                risk_neutral_probability: None,
                theoretical_premium: None,
                absolute_difference: None,
                percentage_difference: None,
                status: "ERROR".to_string(),
                error_message: Some("Invalid or missing numerical values (Spot, Strike, Volatility, or Days).".to_string()),
            });
        }
    }

    (
        StatusCode::OK,
        Json(ProcessDatasetResponse {
            total_records: success_count + error_count,
            success_records: success_count,
            error_records: error_count,
            rows,
        }),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Sample Dataset Downloads
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct DownloadQuery {
    #[serde(rename = "type")]
    pub file_type: String,
}

/// GET /api/download/sample
/// Serves the requested sample CSV file as an attachment
async fn download_sample_handler(Query(query): Query<DownloadQuery>) -> Response {
    let (filename, rel_path) = match query.file_type.to_lowercase().as_str() {
        "ce" => ("nifty50_ce_sample.csv", "data/ce_sample.csv"),
        "pe" => ("nifty50_pe_sample.csv", "data/pe_sample.csv"),
        "template" => ("nifty50_options_template.csv", "data/options_template.csv"),
        _ => ("nifty50_combined_options_sample.csv", "data/combined_options_sample.csv"),
    };

    match fs::read_to_string(rel_path) {
        Ok(content) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, HeaderValue::from_static("text/csv")),
                (
                    header::CONTENT_DISPOSITION,
                    HeaderValue::from_str(&format!("attachment; filename=\"{}\"", filename)).unwrap(),
                ),
            ],
            content,
        )
            .into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            format!("File '{}' not found: {}", rel_path, e),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Single Option Pricing API
// ---------------------------------------------------------------------------

/// GET /api/price
async fn price_handler(Query(query): Query<PricingQuery>) -> Response {
    let option_type = match query.option_type.trim().to_uppercase().as_str() {
        "CE" | "CALL" => OptionType::CE,
        "PE" | "PUT" => OptionType::PE,
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Invalid option type '{}'. Must be 'CE' or 'PE'.", other),
                }),
            )
                .into_response();
        }
    };

    let volatility = if query.volatility > 1.0 {
        query.volatility / 100.0
    } else {
        query.volatility
    };

    let risk_free_rate = match query.risk_free_rate {
        Some(r) if r > 1.0 => r / 100.0,
        Some(r) => r,
        None => 0.065,
    };

    let dividend_yield = match query.dividend_yield {
        Some(q) if q > 1.0 => q / 100.0,
        Some(q) => q,
        None => 0.012,
    };

    let steps = query.steps.unwrap_or(50);

    match calculate_crr(
        query.spot,
        query.strike,
        volatility,
        query.days_to_expiry,
        risk_free_rate,
        dividend_yield,
        steps,
        option_type,
        query.actual_premium,
    ) {
        Ok(res) => (StatusCode::OK, Json(res)).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })).into_response(),
    }
}

/// Serve index.html
async fn serve_index() -> impl IntoResponse {
    let index_path = Path::new("index.html");
    match fs::read_to_string(index_path) {
        Ok(content) => Html(content).into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Html("<h1>index.html not found</h1><p>Please ensure index.html exists in the project root.</p>"),
        )
            .into_response(),
    }
}

/// Build Axum Router
pub fn build_app() -> Router {
    Router::new()
        .route("/", get(serve_index))
        .route("/index.html", get(serve_index))
        .route("/api/price", get(price_handler))
        .route("/api/option-chain", get(option_chain_handler))
        .route("/api/process-dataset", post(process_dataset_handler))
        .route("/api/download/sample", get(download_sample_handler))
        .layer(CorsLayer::permissive())
}

#[tokio::main]
async fn main() {
    let app = build_app();
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));

    println!("============================================================");
    println!("  NIFTY 50 Options Pricing and Analytics Dashboard          ");
    println!("============================================================");
    println!("  Backend Engine: Rust + Axum + Tokio");
    println!("  Server Address: http://127.0.0.1:3000");
    println!("  API Endpoints:");
    println!("    - GET  /api/price           : CRR Binomial Pricing Endpoint");
    println!("    - GET  /api/option-chain    : NIFTY 50 Option Chain Ladder");
    println!("    - POST /api/process-dataset : Upload & Automatic Batch Calculation");
    println!("    - GET  /api/download/sample : Download Sample Datasets & Template");
    println!("  Dashboard UI:   http://127.0.0.1:3000/index.html");
    println!("============================================================");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind TCP listener to 127.0.0.1:3000");

    axum::serve(listener, app)
        .await
        .expect("Server encountered a fatal error during execution");
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_option_calculation() {
        let res = calculate_crr(
            24530.90,
            24500.0,
            0.14,
            14.0,
            0.065,
            0.012,
            50,
            OptionType::CE,
            Some(275.50),
        )
        .expect("CE calculation should succeed");

        assert_eq!(res.option_type, "CE");
        assert!(res.theoretical_premium > 250.0 && res.theoretical_premium < 350.0);
        assert!(res.up_factor > 1.0);
        assert!(res.down_factor < 1.0);
        assert!((res.up_factor * res.down_factor - 1.0).abs() < 1e-4);
        assert!(res.risk_neutral_probability > 0.0 && res.risk_neutral_probability < 1.0);
        assert!(res.absolute_difference.is_some());
    }

    #[test]
    fn test_put_option_calculation() {
        let res = calculate_crr(
            24530.90,
            24500.0,
            0.14,
            14.0,
            0.065,
            0.012,
            50,
            OptionType::PE,
            Some(212.00),
        )
        .expect("PE calculation should succeed");

        assert_eq!(res.option_type, "PE");
        assert!(res.theoretical_premium > 180.0 && res.theoretical_premium < 270.0);
        assert!(res.up_factor > 1.0);
        assert!(res.down_factor < 1.0);
        assert!(res.risk_neutral_probability > 0.0 && res.risk_neutral_probability < 1.0);
    }

    #[test]
    fn test_different_strike_prices() {
        let itm = calculate_crr(24530.90, 24000.0, 0.14, 14.0, 0.065, 0.012, 50, OptionType::CE, None).unwrap();
        let otm = calculate_crr(24530.90, 25000.0, 0.14, 14.0, 0.065, 0.012, 50, OptionType::CE, None).unwrap();
        assert!(itm.theoretical_premium > otm.theoretical_premium);
    }

    #[test]
    fn test_invalid_input_handling() {
        assert!(calculate_crr(-100.0, 24500.0, 0.14, 14.0, 0.065, 0.012, 50, OptionType::CE, None).is_err());
        assert!(calculate_crr(24500.0, 0.0, 0.14, 14.0, 0.065, 0.012, 50, OptionType::CE, None).is_err());
        assert!(calculate_crr(24500.0, 24500.0, 0.0, 14.0, 0.065, 0.012, 50, OptionType::CE, None).is_err());
        assert!(calculate_crr(24500.0, 24500.0, 0.14, -5.0, 0.065, 0.012, 50, OptionType::CE, None).is_err());
    }

    #[test]
    fn test_premium_difference() {
        let res = calculate_crr(24500.0, 24500.0, 0.15, 30.0, 0.065, 0.012, 50, OptionType::CE, Some(400.0)).unwrap();
        let expected_abs = (res.theoretical_premium - 400.0).abs();
        let abs_diff = res.absolute_difference.unwrap();
        assert!((abs_diff - expected_abs).abs() < 1e-2);
    }
}
