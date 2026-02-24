"""
Google GenAI & Anthropic Claude to OpenAI-Compatible API Proxy
Supports Gemini and Claude on Vertex AI
Includes Tool Calling (Function Calling) support
"""
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from anthropic import AnthropicVertex

# Config and models
from config import (
    MASTER_KEY,
    GOOGLE_PROJECT,
    GOOGLE_LOCATION,
    CLAUDE_LOCATION,
    GEMINI_MODEL_MAPPING,
    CLAUDE_MODEL_MAPPING,
    is_claude_model,
)
from models import (
    ChatCompletionRequest,
    ResponseRequest,
    ModelObject,
    ModelsResponse,
)

# Handlers
from handlers.gemini import handle_gemini_request, set_gemini_client
from handlers.claude import handle_claude_request, set_claude_client
from handlers.responses import handle_responses_request


# ============================================================================
# Initialize Clients
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application life cycle management"""
    print("=" * 60)
    print("Initializing AI Clients...")
    print(f"  Project: {GOOGLE_PROJECT}")
    print(f"  Location: {GOOGLE_LOCATION}")
    print("=" * 60)
    
    # Initialize Gemini Client
    print("\n[1/2] Initializing Google GenAI Client (Vertex AI mode)...")
    gemini_client = genai.Client(
        vertexai=True,
        project=GOOGLE_PROJECT,
        location=GOOGLE_LOCATION,
    )
    set_gemini_client(gemini_client)
    print("  ✅ Gemini Client initialized!")
    
    # Initialize Claude Client
    print(f"\n[2/2] Initializing Anthropic Claude Client (Vertex AI mode)...")
    print(f"  Claude Region: {CLAUDE_LOCATION}")
    claude_client = AnthropicVertex(
        project_id=GOOGLE_PROJECT,
        region=CLAUDE_LOCATION,
    )
    set_claude_client(claude_client)
    print("  ✅ Claude Client initialized!")
    
    print("\n" + "=" * 60)
    print("All clients initialized successfully!")
    print(f"  Gemini models: {len(GEMINI_MODEL_MAPPING)}")
    print(f"  Claude models: {len(CLAUDE_MODEL_MAPPING)}")
    print("  Tool calling: ✅ Supported")
    print("=" * 60 + "\n")
    
    yield
    
    print("Shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Vertex AI to OpenAI Proxy",
    description="OpenAI-compatible proxy supporting Gemini and Claude on Vertex AI, including tool calling and Responses API support",
    version="2.2.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Authentication
# ============================================================================

def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """Verify API Key"""
    if not MASTER_KEY:
        return True
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    
    token = authorization[7:]
    if token != MASTER_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Vertex AI to OpenAI Proxy",
        "status": "running",
        "version": "2.2.0",
        "supported_providers": ["google/gemini", "anthropic/claude"],
        "features": ["chat", "streaming", "tool_calling", "structured_output", "responses_api"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """List available models"""
    verify_api_key(authorization)
    
    models = []
    seen = set()
    
    # Add Gemini models
    for model_name in GEMINI_MODEL_MAPPING.keys():
        if model_name not in seen:
            models.append(ModelObject(id=model_name, owned_by="google"))
            seen.add(model_name)
    
    # Add Claude models
    for model_name in CLAUDE_MODEL_MAPPING.keys():
        if model_name not in seen:
            models.append(ModelObject(id=model_name, owned_by="anthropic"))
            seen.add(model_name)
    
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    """Chat Completions API - Automatically routes to Gemini or Claude, supports tool calling"""
    verify_api_key(authorization)
    
    model_name = request.model
    
    # Route to different processors based on model name
    if is_claude_model(model_name):
        return await handle_claude_request(request, model_name)
    else:
        return await handle_gemini_request(request, model_name)


# ============================================================================
# Responses API (New Standard)
# ============================================================================

@app.post("/v1/responses")
async def create_response(
    request: ResponseRequest,
    authorization: Optional[str] = Header(None),
):
    """Responses API - OpenAI's new standard, automatically routes to Gemini or Claude"""
    verify_api_key(authorization)
    return await handle_responses_request(request, authorization)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)