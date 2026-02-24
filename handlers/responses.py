"""
Responses API Handler Module
OpenAI New Standard /v1/responses Endpoint
"""
import json
import time
import uuid
from typing import AsyncGenerator, List, Optional, Union

from config import is_claude_model
from models import (
    ResponseRequest,
    ResponseInputItem,
    Response,
    ResponseOutputItem,
    ResponseOutputContent,
    ResponseUsage,
    ChatMessage,
    ChatCompletionRequest,
    ToolCall,
    Tool,
)


def convert_response_request_to_chat(request: ResponseRequest) -> ChatCompletionRequest:
    """
    Convert Responses API request to Chat Completions internal format
    """
    messages: List[ChatMessage] = []
    
    # Add instructions as system message
    if request.instructions:
        messages.append(ChatMessage(role="system", content=request.instructions))
    
    # Handle input
    if isinstance(request.input, str):
        # Simple string input -> user message
        messages.append(ChatMessage(role="user", content=request.input))
    elif isinstance(request.input, list):
        # Input item array
        for item in request.input:
            if isinstance(item, dict):
                # Convert from dict
                role = item.get("role", "user")
                content = item.get("content")
                tool_calls = item.get("tool_calls")
                tool_call_id = item.get("tool_call_id")
                name = item.get("name")
            else:
                # ResponseInputItem object
                role = item.role or "user"
                content = item.content
                tool_calls = item.tool_calls
                tool_call_id = item.tool_call_id
                name = item.name
            
            messages.append(ChatMessage(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                name=name,
            ))
    
    # Handle text.format conversion to response_format
    response_format = None
    if request.text and request.text.format:
        from models import ResponseFormat
        fmt = request.text.format
        if fmt.type == "json_schema":
            response_format = ResponseFormat(
                type="json_schema",
                json_schema=fmt.json_schema
            )
        elif fmt.type == "json_object":
            response_format = ResponseFormat(type="json_object")
    
    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice,
        response_format=response_format,
    )


def convert_chat_response_to_response(
    chat_response: dict,
    request: ResponseRequest,
) -> Response:
    """
    Convert Chat Completions response to Responses API format
    """
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    
    # Extract message
    choice = chat_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    
    # Build output
    output_items: List[ResponseOutputItem] = []
    
    # Check for tool calls
    tool_calls = message.get("tool_calls", [])
    content = message.get("content")
    
    # Add message output
    output_content: List[ResponseOutputContent] = []
    
    if content:
        output_content.append(ResponseOutputContent(
            type="output_text",
            text=content,
            annotations=[],
        ))
    
    # Add tool calls as separate output items
    if tool_calls:
        for tc in tool_calls:
            output_items.append(ResponseOutputItem(
                type="function_call",
                id=tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                status="completed",
                role=None,
                content=[ResponseOutputContent(
                    type="function_call",
                    id=tc.get("id"),
                    name=tc.get("function", {}).get("name"),
                    arguments=tc.get("function", {}).get("arguments"),
                )],
            ))
    
    # Add main message output (if there is text content)
    if output_content:
        output_items.insert(0, ResponseOutputItem(
            type="message",
            id=f"msg_{uuid.uuid4().hex}",
            status="completed",
            role="assistant",
            content=output_content,
        ))
    
    # If there are no outputs, add an empty message
    if not output_items:
        output_items.append(ResponseOutputItem(
            type="message",
            id=f"msg_{uuid.uuid4().hex}",
            status="completed",
            role="assistant",
            content=[],
        ))
    
    # Build usage
    usage_data = chat_response.get("usage", {})
    usage = ResponseUsage(
        input_tokens=usage_data.get("prompt_tokens", 0),
        output_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    )
    
    # Build tool list (convert to dict)
    tools_dict = []
    if request.tools:
        for tool in request.tools:
            tools_dict.append(tool.model_dump())
    
    return Response(
        id=response_id,
        created_at=created_at,
        completed_at=int(time.time()),
        status="completed",
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        model=chat_response.get("model", request.model),
        output=output_items,
        parallel_tool_calls=request.parallel_tool_calls or True,
        temperature=request.temperature or 1.0,
        tool_choice=request.tool_choice if isinstance(request.tool_choice, str) else "auto",
        tools=tools_dict,
        top_p=request.top_p or 1.0,
        truncation=request.truncation or "disabled",
        usage=usage,
        user=request.user,
        metadata=request.metadata or {},
    )


async def stream_responses_format(
    stream_generator: AsyncGenerator[str, None],
    request: ResponseRequest,
) -> AsyncGenerator[str, None]:
    """
    Convert Chat Completions streaming response to Responses API streaming format
    """
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    
    # Send initial event
    initial_event = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": request.model,
        }
    }
    yield f"data: {json.dumps(initial_event)}\n\n"
    
    # Forward content
    async for chunk in stream_generator:
        if chunk.startswith("data: "):
            data_str = chunk[6:].strip()
            if data_str == "[DONE]":
                # Send completion event
                done_event = {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "completed_at": int(time.time()),
                    }
                }
                yield f"data: {json.dumps(done_event)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    
                    if delta.get("content"):
                        # Text increment
                        text_event = {
                            "type": "response.output_text.delta",
                            "delta": delta.get("content"),
                        }
                        yield f"data: {json.dumps(text_event)}\n\n"
                    
                    if delta.get("tool_calls"):
                        # Tool call increment
                        for tc in delta.get("tool_calls", []):
                            tool_event = {
                                "type": "response.function_call.delta",
                                "tool_call": tc,
                            }
                            yield f"data: {json.dumps(tool_event)}\n\n"
                except json.JSONDecodeError:
                    pass


async def handle_responses_request(request: ResponseRequest, authorization: str):
    """
    Handle /v1/responses request
    """
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse
    
    # Convert to Chat Completions format
    chat_request = convert_response_request_to_chat(request)
    
    # Route based on model
    if is_claude_model(request.model):
        from handlers.claude import handle_claude_request
        result = await handle_claude_request(chat_request, request.model)
    else:
        from handlers.gemini import handle_gemini_request
        result = await handle_gemini_request(chat_request, request.model)
    
    # Handle streaming response
    if request.stream:
        if isinstance(result, StreamingResponse):
            # Need to convert streaming format
            return StreamingResponse(
                stream_responses_format(result.body_iterator, request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                }
            )
        return result
    
    # Non-streaming response - convert format
    if isinstance(result, dict):
        return convert_chat_response_to_response(result, request)
    elif hasattr(result, 'body'):
        # JSONResponse
        chat_response = json.loads(result.body)
        return convert_chat_response_to_response(chat_response, request)
    else:
        return convert_chat_response_to_response(result, request)
