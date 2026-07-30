use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use serde_json::{json, Value};
use std::{env, net::SocketAddr, sync::Arc};
use tower_http::cors::CorsLayer;

// --- App State ---
#[derive(Clone)]
struct AppState {
    http_client: Client,
    gemini_api_key: String,
}

// --- Main Server ---
#[tokio::main]
async fn main() {
    // Load .env if it exists (for local testing)
    let _ = dotenvy::dotenv();
    
    // Fail fast on boot if the key is missing
    let gemini_api_key = env::var("GEMINI_API_KEY")
        .expect("❌ GEMINI_API_KEY environment variable is required");
    
    let port = env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    
    let state = Arc::new(AppState {
        http_client: Client::new(),
        gemini_api_key,
    });

    // Native CORS support out of the box
    let app = Router::new()
        .route("/api/health", get(health_check))
        .route("/api/negotiate/start", post(handle_negotiation))
        .layer(CorsLayer::permissive()) 
        .with_state(state);

    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse().unwrap();
    println!("🚀 Server running on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// --- Route Handlers ---
async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

async fn handle_negotiation(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    // Build a prompt from the request payload
    let prompt = match &payload {
        Value::Object(map) => {
            let item = map.get("item")
                .and_then(|v| v.as_str())
                .unwrap_or("item");
            let initial_price = map.get("initialPrice")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let target_price = map.get("targetPrice")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            
            format!(
                "I am negotiating to buy a {} currently priced at ${}, and I want to negotiate it down to ${}. Help me with a negotiation strategy.",
                item, initial_price, target_price
            )
        }
        _ => "Please provide negotiation details with item, initialPrice, and targetPrice.".to_string(),
    };

    println!("📨 Sending prompt to Gemini: {}", prompt);

    // Use gemini-2.0-flash-lite which is available and performant
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={}",
        state.gemini_api_key
    );

    let gemini_req = json!({
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    });

    let res = match state.http_client.post(&url).json(&gemini_req).send().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ Failed to reach Gemini API: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"reply": "Failed to reach Gemini API"}))).into_response();
        }
    };

    let status = res.status();
    let text = match res.text().await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("❌ Failed to read response body: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"reply": "Failed to read API response"}))).into_response();
        }
    };

    println!("📩 Gemini API Response (Status: {}): {}", status, text);

    match serde_json::from_str::<Value>(&text) {
        Ok(gemini_data) => {
            // Try to extract the text from the response
            let reply_text = gemini_data
                .get("candidates")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|c| c.get("content"))
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
                .and_then(|arr| arr.first())
                .and_then(|p| p.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or_else(|| {
                    eprintln!("❌ Could not extract text from response: {}", gemini_data);
                    "No response generated from Gemini API"
                });

            println!("✅ Extracted reply: {}", reply_text);
            (StatusCode::OK, Json(json!({"reply": reply_text}))).into_response()
        }
        Err(e) => {
            eprintln!("❌ Failed to parse Gemini response as JSON: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"reply": "Failed to parse Gemini response"}))).into_response()
        }
    }
}

