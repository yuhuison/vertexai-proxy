"""
Gemini Request Handler Module
"""
import base64
import json
import time
import uuid
from typing import AsyncGenerator, List

from google import genai
from google.genai.types import GenerateContentConfig, Content

from config import GEMINI_MODEL_MAPPING, SAFETY_SETTINGS
from models import ChatCompletionRequest, ToolCall
from converters.messages import convert_messages_to_genai
from converters.tools import (
    convert_tools_to_gemini,
    convert_tool_choice_to_gemini,
    convert_gemini_function_call,
)
from thought_signature_cache import store_thought_signature


# Gemini client - Set by main.py
gemini_client: genai.Client = None


def set_gemini_client(client: genai.Client):
    """Set Gemini client"""
    global gemini_client
    gemini_client = client


async def handle_gemini_request(request: ChatCompletionRequest, model_name: str):
    """Handle Gemini request"""
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse
    
    genai_model = GEMINI_MODEL_MAPPING.get(model_name, model_name)
    
    system_instruction, contents = convert_messages_to_genai(request.messages)
    
    # Build base configuration
    config_kwargs = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_output_tokens": request.max_tokens,
        "stop_sequences": request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
        "safety_settings": SAFETY_SETTINGS,
        "system_instruction": system_instruction,
    }
    
    # Handle response_format (structured output)
    if request.response_format:
        if request.response_format.type == "json_schema":
            config_kwargs["response_mime_type"] = "application/json"
            if request.response_format.json_schema:
                schema = request.response_format.json_schema
                if "schema" in schema:
                    config_kwargs["response_schema"] = schema["schema"]
                else:
                    config_kwargs["response_schema"] = schema
        elif request.response_format.type == "json_object":
            config_kwargs["response_mime_type"] = "application/json"
    
    # Handle tool calling
    if request.tools:
        config_kwargs["tools"] = [convert_tools_to_gemini(request.tools)]
        
        # Handle tool_choice
        if request.tool_choice:
            tool_config = convert_tool_choice_to_gemini(request.tool_choice)
            if tool_config:
                config_kwargs["tool_config"] = tool_config
    
    config = GenerateContentConfig(**config_kwargs)
    
    try:
        if request.stream:
            return StreamingResponse(
                stream_gemini_response(genai_model, contents, config, model_name),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                }
            )
        else:
            response = gemini_client.models.generate_content(
                model=genai_model,
                contents=contents,
                config=config,
            )
            return create_gemini_response(response, model_name)
            
    except Exception as e:
        print(f"Gemini error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_gemini_response(
    model: str,
    contents: List[Content],
    config: GenerateContentConfig,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """Generate Gemini streaming response - Supports tool calling"""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    
    try:
        response_stream = gemini_client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        
        accumulated_tool_calls = []
        
        for chunk in response_stream:
            # Check for function_call
            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # ⭐ Each Part independently checks thought_signature
                        # Parallel calling: Only the first function_call has signature
                        # Sequential calling: Each step has its own signature
                        part_thought_signature = None
                        if hasattr(part, 'thought_signature') and part.thought_signature:
                            ts = part.thought_signature
                            if isinstance(ts, bytes):
                                part_thought_signature = base64.b64encode(ts).decode('utf-8')
                            else:
                                part_thought_signature = str(ts)
                        
                        # Pass thought_signature
                        tool_call = convert_gemini_function_call(part.function_call, part_thought_signature)
                        accumulated_tool_calls.append(tool_call)
                        
                        # ⭐ Store this Part's thought_signature (if any)
                        if part_thought_signature:
                            try:
                                ts_bytes = base64.b64decode(part_thought_signature)
                                store_thought_signature(tool_call.id, ts_bytes)
                            except Exception:
                                pass
                        
                        # Send tool call chunk (Send only OpenAI standard format fields, not including thought_signature)
                        tool_call_data = {
                            "index": len(accumulated_tool_calls) - 1,
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                            # Note: thought_signature is an internal Gemini field, not sent to the client
                            # Already stored via store_thought_signature()
                        }
                        
                        delta = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [tool_call_data]
                                },
                                "finish_reason": None,
                            }]
                        }
                        yield f"data: {json.dumps(delta)}\n\n"
                    elif hasattr(part, 'text') and part.text:
                        delta = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": part.text},
                                "finish_reason": None,
                            }]
                        }
                        yield f"data: {json.dumps(delta)}\n\n"
        
        # Determine finish_reason
        finish_reason = "tool_calls" if accumulated_tool_calls else "stop"
        
        final_delta = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }]
        }
        yield f"data: {json.dumps(final_delta)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_data = {"error": {"message": str(e), "type": "api_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"


def create_gemini_response(response, model_name: str) -> dict:
    """Create Gemini OpenAI format response - Supports tool calling"""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    
    content = ""
    tool_calls = []
    
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                # ⭐ Each Part independently checks thought_signature
                part_thought_signature = None
                if hasattr(part, 'thought_signature') and part.thought_signature:
                    ts = part.thought_signature
                    if isinstance(ts, bytes):
                        part_thought_signature = base64.b64encode(ts).decode('utf-8')
                    else:
                        part_thought_signature = str(ts)
                
                # Pass thought_signature
                tool_call = convert_gemini_function_call(part.function_call, part_thought_signature)
                tool_call_dict = tool_call.model_dump()
                
                # ⭐ Store this Part's thought_signature (if any)
                if part_thought_signature:
                    try:
                        ts_bytes = base64.b64decode(part_thought_signature)
                        store_thought_signature(tool_call.id, ts_bytes)
                    except Exception:
                        pass
                
                # Remove thought_signature field (Do not send to client)
                if "thought_signature" in tool_call_dict:
                    del tool_call_dict["thought_signature"]
                
                tool_calls.append(tool_call_dict)
            elif hasattr(part, 'text') and part.text:
                content += part.text
    
    # Determine finish_reason
    finish_reason = "tool_calls" if tool_calls else "stop"
    
    # Build message
    message = {"role": "assistant"}
    if content:
        message["content"] = content
    else:
        message["content"] = None
    
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    # Calculate tokens (Estimated)
    prompt_tokens = sum(len(str(c)) for c in response.candidates[0].content.parts) // 4 if response.candidates else 0
    completion_tokens = len(content) // 4 + len(json.dumps(tool_calls)) // 4
    
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }
