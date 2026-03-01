"""
Pydantic Models - OpenAI-Compatible Request/Response Formats
Includes Tool Calling (Function Calling) support
"""
import time
from typing import List, Optional, Union

from pydantic import BaseModel, Field


# ============================================================================
# Tool Calling Related Models (OpenAI Format)
# ============================================================================

class ToolFunction(BaseModel):
    """Tool function definition"""
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None  # JSON Schema


class Tool(BaseModel):
    """Tool definition"""
    type: str = "function"
    function: ToolFunction


class FunctionCall(BaseModel):
    """Function call content"""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call
    Note: Gemini 3.0+ requires thought_signature field for multi-turn tool calling
    """
    id: str
    type: str = "function"
    function: FunctionCall
    # Gemini 3.0+ thought_signature - Must be retained and passed back in multi-turn conversations
    thought_signature: Optional[str] = None


# ============================================================================
# Message Models
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message - Supports tool calling"""
    role: str  # "user" | "assistant" | "system" | "tool"
    content: Optional[Union[str, List[dict]]] = None
    name: Optional[str] = None
    # Tool calling related
    tool_calls: Optional[List[ToolCall]] = None  # Used by assistant role
    tool_call_id: Optional[str] = None  # Used by tool role
    # Gemini 3.0+ thought_signature - Must be retained in multi-turn conversations
    thought_signature: Optional[str] = None


# ============================================================================
# Response Format
# ============================================================================

class ResponseFormat(BaseModel):
    """OpenAI response_format structure"""
    type: str  # "json_schema" or "json_object" or "text"
    json_schema: Optional[dict] = None  # Used when type="json_schema"


# ============================================================================
# Request Models
# ============================================================================

class ChatCompletionRequest(BaseModel):
    """Chat completion request - Supports tool calling"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    response_format: Optional[ResponseFormat] = None
    # Tool calling related
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, dict]] = None  # "auto" | "none" | "required" | {"type": "function", "function": {"name": "xxx"}}


# ============================================================================
# Response Models
# ============================================================================

class ModelObject(BaseModel):
    """Model object"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "google"


class ModelsResponse(BaseModel):
    """Model list response"""
    object: str = "list"
    data: List[ModelObject]
