"""
Configuration Module - Model Mappings, Environment Variables, Security Settings
"""
import os

from google.genai.types import (
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)


# ============================================================================
# Model Mapping
# ============================================================================

# Gemini Model Mapping: OpenAI-style name -> Google GenAI model name
GEMINI_MODEL_MAPPING = {
    # Gemini 3 series
    "google/gemini-3-pro-preview": "gemini-3-pro-preview",
    "google/gemini-3-flash-preview": "gemini-3-flash-preview",
    "google/gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    
    # Gemini 2.5 series
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-lite-preview-09-2025": "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025": "gemini-2.5-flash-lite-preview-09-2025",
}

# Claude Model Mapping: OpenAI-style name -> Vertex AI Model ID
CLAUDE_MODEL_MAPPING = {
    # Claude 4.5 series
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4-5@20250929",
    "anthropic/claude-opus-4.5": "claude-opus-4-5@20251101",
    "claude-sonnet-4.5": "claude-sonnet-4-5@20250929",
    "claude-opus-4.5": "claude-opus-4-5@20251101",
    # Claude Haiku series
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5@20251001",
    "claude-haiku-4.5": "claude-haiku-4-5@20251001",
}

# Merge all model mappings
ALL_MODEL_MAPPING = {**GEMINI_MODEL_MAPPING, **CLAUDE_MODEL_MAPPING}


# ============================================================================
# Environment Variables Configuration
# ============================================================================

# API Key (Read from environment variables, used for client authentication)
MASTER_KEY = os.environ.get("MASTER_KEY", "")

# Google Cloud Project Configuration (Vertex AI mode)
GOOGLE_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

# Claude Region Configuration
# Note: Only Claude Sonnet 4.5+ supports global endpoint
CLAUDE_LOCATION = os.environ.get("CLAUDE_LOCATION", "global")


# ============================================================================
# Safety Settings
# ============================================================================

# Gemini Safety Settings - All set to OFF
SAFETY_SETTINGS = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
]


# ============================================================================
# Helper Functions
# ============================================================================

def is_claude_model(model_name: str) -> bool:
    """Check if it is a Claude model"""
    return model_name in CLAUDE_MODEL_MAPPING or model_name.startswith(("anthropic/", "claude"))


def is_gemini_model(model_name: str) -> bool:
    """Check if it is a Gemini model"""
    return model_name in GEMINI_MODEL_MAPPING or model_name.startswith(("google/", "gemini"))
