"""
Claude Request Handler Module
"""
import json
import time
import traceback
import uuid
from typing import AsyncGenerator

from anthropic import AnthropicVertex

from config import CLAUDE_MODEL_MAPPING
from models import ChatCompletionRequest, ToolCall
from converters.messages import convert_messages_to_claude
from converters.tools import (
    convert_tools_to_claude,
    convert_tool_choice_to_claude,
    convert_claude_tool_use,
)


# Claude client - Set by main.py
claude_client: AnthropicVertex = None


def set_claude_client(client: AnthropicVertex):
    """Set Claude client"""
    global claude_client
    claude_client = client


async def handle_claude_request(request: ChatCompletionRequest, model_name: str):
    """Handle Claude request"""
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse
    
    claude_model = CLAUDE_MODEL_MAPPING.get(model_name, model_name)
    
    system_prompt, messages = convert_messages_to_claude(request.messages)
    
    # Build request parameters
    kwargs = {
        "model": claude_model,
        "max_tokens": request.max_tokens or 4096,
        "messages": messages,
    }
    
    if system_prompt:
        kwargs["system"] = system_prompt
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.stop:
        kwargs["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]
    
    # Handle tool calling
    if request.tools:
        kwargs["tools"] = convert_tools_to_claude(request.tools)
        
        # Handle tool_choice
        if request.tool_choice:
            tool_choice = convert_tool_choice_to_claude(request.tool_choice)
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
    
    # Handle response_format (Claude implements structured output using output_config)
    # Note: If there are tools, output_config should not be used
    if request.response_format and not request.tools:
        if request.response_format.type == "json_schema" and request.response_format.json_schema:
            schema = request.response_format.json_schema
            output_schema = schema.get("schema", schema)
            
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": output_schema,
                }
            }
        elif request.response_format.type == "json_object":
            json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only. Do not include any text outside the JSON object."
            if kwargs.get("system"):
                kwargs["system"] += json_instruction
            else:
                kwargs["system"] = json_instruction.strip()
    
    try:
        if request.stream:
            return StreamingResponse(
                stream_claude_response(kwargs, model_name),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                }
            )
        else:
            response = claude_client.messages.create(**kwargs)
            return create_claude_response(response, model_name)
            
    except Exception as e:
        print(f"Claude error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def stream_claude_response(kwargs: dict, model_name: str) -> AsyncGenerator[str, None]:
    """Generate Claude streaming response - Supports tool calling"""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    
    try:
        accumulated_tool_calls = []
        current_tool_call = None
        tool_call_index = 0
        
        with claude_client.messages.stream(**kwargs) as stream:
            for event in stream:
                # Handle different types of events
                if hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        # Check if it is a tool_use block
                        if hasattr(event, 'content_block') and hasattr(event.content_block, 'type'):
                            if event.content_block.type == 'tool_use':
                                current_tool_call = {
                                    "id": event.content_block.id,
                                    "name": event.content_block.name,
                                    "input": ""
                                }
                                # Send tool calling start
                                delta = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": tool_call_index,
                                                "id": current_tool_call["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": current_tool_call["name"],
                                                    "arguments": ""
                                                }
                                            }]
                                        },
                                        "finish_reason": None,
                                    }]
                                }
                                yield f"data: {json.dumps(delta)}\n\n"
                    
                    elif event.type == 'content_block_delta':
                        if hasattr(event, 'delta'):
                            if hasattr(event.delta, 'type'):
                                if event.delta.type == 'input_json_delta':
                                    # Tool input increment
                                    if current_tool_call:
                                        partial_json = event.delta.partial_json
                                        delta = {
                                            "id": request_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "tool_calls": [{
                                                        "index": tool_call_index,
                                                        "function": {
                                                            "arguments": partial_json
                                                        }
                                                    }]
                                                },
                                                "finish_reason": None,
                                            }]
                                        }
                                        yield f"data: {json.dumps(delta)}\n\n"
                                elif event.delta.type == 'text_delta':
                                    # Text increment
                                    delta = {
                                        "id": request_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": event.delta.text},
                                            "finish_reason": None,
                                        }]
                                    }
                                    yield f"data: {json.dumps(delta)}\n\n"
                    
                    elif event.type == 'content_block_stop':
                        if current_tool_call:
                            accumulated_tool_calls.append(current_tool_call)
                            current_tool_call = None
                            tool_call_index += 1
                
                # Compatible with old text_stream method
                elif hasattr(event, 'text'):
                    delta = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": event.text},
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


def create_claude_response(response, model_name: str) -> dict:
    """Create Claude OpenAI format response - Supports tool calling"""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    
    content = ""
    tool_calls = []

    # Extract content and tool calls
    if not response.content:
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    for block in response.content:
        if hasattr(block, 'type'):
            if block.type == 'text':
                content += block.text
            elif block.type == 'tool_use':
                tool_call = convert_claude_tool_use(block)
                tool_calls.append(tool_call.model_dump())
        elif hasattr(block, 'text'):
            content += block.text
    
    # Determine finish_reason
    # Claude's stop_reason: "end_turn", "tool_use", "max_tokens", "stop_sequence"
    if response.stop_reason == "tool_use" or tool_calls:
        finish_reason = "tool_calls"
    elif response.stop_reason == "max_tokens":
        finish_reason = "length"
    else:
        finish_reason = "stop"
    
    # Build message
    message = {"role": "assistant"}
    if content:
        message["content"] = content
    else:
        message["content"] = None
    
    if tool_calls:
        message["tool_calls"] = tool_calls
    
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
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
    }
