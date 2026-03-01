# Vertex AI to OpenAI Proxy

[![Version](https://img.shields.io/badge/version-3.0.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.11+-green)]()

OpenAI-compatible proxy service supporting Google Gemini and Anthropic Claude on Vertex AI.

## Features

| Feature | Status |
|------|------|
| Chat Completions API (`/v1/chat/completions`) | ✅ |
| Streaming Responses (SSE) | ✅ |
| Tool Calling (Function Calling) | ✅ |
| Structured Output (JSON Schema) | ✅ |
| Multimodal (Image Input) | ✅ |
| Gemini 3.0+ `thought_signature` | ✅ |

## Project Structure

```
vertexai-proxy/
├── main.py                 # FastAPI application entry point and routing
├── config.py               # Configuration, model mappings, environment variables
├── models.py               # Pydantic data models
├── Dockerfile              # Docker build configuration
├── test_cors.py            # Test script
│
├── handlers/               # Request handlers
│   ├── __init__.py
│   ├── gemini.py           # Gemini model handling
│   └── claude.py           # Claude model handling
│
└── converters/             # Format converters
    ├── __init__.py
    ├── messages.py         # Message format conversion
    └── tools.py            # Tool format conversion
```

## Architecture Design

### Request Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   main.py   │────▶│  handlers/  │────▶│  Vertex AI  │
│  (OpenAI)   │     │  (Routing)  │     │ (Handlers)  │     │  (Gemini/   │
│             │◀────│             │◀────│             │◀────│   Claude)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  models.py  │     │ converters/ │
                    │ (Data Models)│     │ (Conversion)│
                    └─────────────┘     └─────────────┘
```

### Module Responsibilities

#### `main.py` - Application Entry Point
- FastAPI application initialization.
- CORS middleware configuration.
- API Key authentication.
- Route definitions (`/v1/chat/completions`, `/v1/models`).
- Client initialization (lifespan).

#### `config.py` - Configuration Center
```python
# Model Mapping
GEMINI_MODEL_MAPPING = { "gemini-3-flash-preview": "gemini-3-flash-preview", ... }
CLAUDE_MODEL_MAPPING = { "claude-sonnet-4.5": "claude-sonnet-4-5@20250929", ... }

# Environment Variables
MASTER_KEY          # API Key
GOOGLE_PROJECT      # GCP Project ID
GOOGLE_LOCATION     # Gemini Region (defaults to global)
CLAUDE_LOCATION     # Claude Region (defaults to global)

# Helper Functions
is_claude_model(name)   # Checks if it's a Claude model
is_gemini_model(name)   # Checks if it's a Gemini model
```

#### `models.py` - Data Models
```python
# Tool Calling
ToolFunction        # Function definition (name, description, parameters)
Tool                # Tool definition (type, function)
FunctionCall        # Function call content (name, arguments)
ToolCall            # Tool call (id, type, function, thought_signature)

# Message
ChatMessage         # Chat message (role, content, tool_calls, thought_signature)

# Request
ChatCompletionRequest   # /v1/chat/completions request
```

#### `handlers/gemini.py` - Gemini Handler
- `handle_gemini_request()` - Main handler function.
- `stream_gemini_response()` - Streaming response generation.
- `create_gemini_response()` - Non-streaming response construction.
- **Important**: Extracts `thought_signature` and returns it to client (required for Gemini 3.0+ multi-turn tool calling).

#### `handlers/claude.py` - Claude Handler
- `handle_claude_request()` - Main handler function.
- `stream_claude_response()` - Streaming response generation.
- `create_claude_response()` - Non-streaming response construction.

#### `converters/messages.py` - Message Conversion
- `convert_messages_to_genai()` - OpenAI → Gemini message format.
- `convert_messages_to_claude()` - OpenAI → Claude message format.
- Handles `tool` role messages (tool response).
- Handles `tool_calls` in `assistant` messages.
- Reads `thought_signature` from client's tool_calls and passes to Gemini.

#### `converters/tools.py` - Tool Conversion
- `convert_tools_to_gemini()` - OpenAI → Gemini tool definition.
- `convert_tools_to_claude()` - OpenAI → Claude tool definition.
- `convert_tool_choice_to_gemini()` - `tool_choice` conversion.
- `convert_tool_choice_to_claude()` - `tool_choice` conversion.
- `convert_gemini_function_call()` - Gemini response → OpenAI `tool_call`.
- `convert_claude_tool_use()` - Claude response → OpenAI `tool_call`.

## Key Implementation Details

### 1. `thought_signature` (Gemini 3.0+)

Gemini 3.0 models return an encrypted `thought_signature` during tool calls, which must be returned as-is in subsequent requests. The proxy extracts it and includes it in the response `tool_calls` — the client is responsible for saving it and passing it back in the next request.

### 2. Tool Calling Flow

```
1. Client sends a request with tools.
2. Proxy converts tool definitions (converters/tools.py).
3. Sent to Gemini/Claude.
4. Model returns function_call / tool_use.
5. Proxy converts it to OpenAI tool_calls format.
6. Client executes the tool and sends a tool message.
7. Proxy converts the tool message to a function_response.
8. Loop until the model returns the final response.
```

## Environment Variables

| Variable | Required | Default Value | Description |
|------|------|--------|------|
| `MASTER_KEY` | No | - | API authentication key |
| `GOOGLE_CLOUD_PROJECT` | Yes | - | GCP Project ID |
| `GOOGLE_CLOUD_LOCATION` | No | `global` | Gemini Region |
| `CLAUDE_LOCATION` | No | `global` | Claude Region |
| `GOOGLE_GENAI_USE_VERTEXAI` | Yes | `true` | Use Vertex AI mode |

## Deployment

### Docker Build
```bash
docker build -t vertex-ai-proxy .
docker run -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=your-project \
  -e MASTER_KEY=your-api-key \
  vertex-ai-proxy
```

### Cloud Run Deployment
```bash
gcloud run deploy vertex-ai-proxy \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated
```

## Testing

```bash
# Run test script
python test_cors.py

# Manual testing
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-flash-preview",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Adding New Models

1. Add mapping in `GEMINI_MODEL_MAPPING` or `CLAUDE_MODEL_MAPPING` in `config.py`.
2. If it's a new provider, you need to:
   - Create a new `handlers/xxx.py`.
   - Create new converter functions.
   - Add routing logic in `main.py`.

## FAQ

### `thought_signature` Error
**Error**: `function call is missing a thought_signature`
**Reason**: `thought_signature` was not passed back during multi-turn tool calling.
**Solution**: Ensure the client saves the `thought_signature` field from the response's `tool_calls` and includes it when sending back the assistant message.

### Model Not Supported
**Error**: `Model not found`
**Solution**: Check the model mapping in `config.py`.

### Region Error
**Error**: `Location not supported`
**Solution**: Some Claude models do not support `global`; use `us-east5` or other supported regions.
