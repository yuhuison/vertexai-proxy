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


# ============================================================================
# Responses API Model (OpenAI New Standard)
# ============================================================================

class ResponseInputItem(BaseModel):
    """Responses API input item"""
    type: str = "message"  # "message" | "item_reference"
    role: Optional[str] = None  # "user" | "assistant" | "system"
    content: Optional[Union[str, List[dict]]] = None
    # Tool calling related
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ResponseTextFormat(BaseModel):
    """Text format configuration"""
    type: str = "text"  # "text" | "json_schema" | "json_object"
    json_schema: Optional[dict] = None


class ResponseTextConfig(BaseModel):
    """Text configuration"""
    format: Optional[ResponseTextFormat] = None


class ResponseRequest(BaseModel):
    """POST /v1/responses request body"""
    model: str
    input: Union[str, List[ResponseInputItem]]  # Text or array of input items
    instructions: Optional[str] = None  # System prompt (independent of input)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, dict]] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_output_tokens: Optional[int] = None
    stream: Optional[bool] = False
    parallel_tool_calls: Optional[bool] = True
    store: Optional[bool] = True
    metadata: Optional[dict] = None
    text: Optional[ResponseTextConfig] = None
    truncation: Optional[str] = "disabled"
    user: Optional[str] = None


class ResponseOutputContent(BaseModel):
    """Response output content"""
    type: str  # "output_text" | "tool_use"
    text: Optional[str] = None
    annotations: Optional[List[dict]] = Field(default_factory=list)
    # Tool calling related
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None


class ResponseOutputItem(BaseModel):
    """Response output item"""
    type: str  # "message" | "function_call"
    id: str
    status: str = "completed"
    role: Optional[str] = None
    content: Optional[List[ResponseOutputContent]] = None


class ResponseUsageDetails(BaseModel):
    """Usage details"""
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    """Usage statistics"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: Optional[dict] = Field(default_factory=lambda: {"cached_tokens": 0})
    output_tokens_details: Optional[dict] = Field(default_factory=lambda: {"reasoning_tokens": 0})


class Response(BaseModel):
    """POST /v1/responses response body"""
    id: str
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: str = "completed"
    completed_at: Optional[int] = None
    error: Optional[dict] = None
    incomplete_details: Optional[dict] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    model: str
    output: List[ResponseOutputItem]
    parallel_tool_calls: bool = True
    previous_response_id: Optional[str] = None
    reasoning: Optional[dict] = Field(default_factory=lambda: {"effort": None, "summary": None})
    store: bool = True
    temperature: float = 1.0
    text: Optional[dict] = Field(default_factory=lambda: {"format": {"type": "text"}})
    tool_choice: Optional[str] = "auto"
    tools: List[dict] = Field(default_factory=list)
    top_p: float = 1.0
    truncation: str = "disabled"
    usage: ResponseUsage
    user: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
