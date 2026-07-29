use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, sync::Arc};
use tower_http::cors::CorsLayer;

// --- App State ---
#[derive(Clone)]
struct AppState {
    http_client: Client,
    gemini_api_key: String,
}

// --- Request/Response Structs ---
#[derive(Deserialize)]
struct NegotiationRequest {
    prompt: String,
}

#[derive(Serialize)]
struct NegotiationResponse {
    reply: String,
}

// --- Gemini REST API Structs ---
#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
}
#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}
#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}
#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContentResponse,
}
#[derive(Deserialize)]
struct GeminiContentResponse {
    parts: Vec<GeminiPartResponse>,
}
#[derive(Deserialize)]
struct GeminiPartResponse {
    text: String,
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
        .route("/api/negotiate", post(handle_negotiation))
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
    Json(payload): Json<NegotiationRequest>,
) -> impl IntoResponse {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={}",
        state.gemini_api_key
    );

    let gemini_req = GeminiRequest {
        contents: vec![GeminiContent {
            parts: vec![GeminiPart { text: payload.prompt }],
        }],
    };

    let res = match state.http_client.post(&url).json(&gemini_req).send().await {
        Ok(r) => r,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to reach Gemini API").into_response(),
    };

    let gemini_data: GeminiResponse = match res.json().await {
        Ok(data) => data,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to parse Gemini response").into_response(),
    };

    let reply_text = gemini_data
        .candidates
        .and_then(|mut c| c.pop())
        .and_then(|mut c| c.content.parts.pop())
        .map(|p| p.text)
        .unwrap_or_else(|| "No response generated".to_string());

    let response = NegotiationResponse { reply: reply_text };
    (StatusCode::OK, Json(response)).into_response()
}