"""
Message Format Conversion - OpenAI Format <-> Gemini/Claude Format
"""
import base64
import json
from typing import List, Optional

from google.genai.types import Content, Part

from models import ChatMessage


# ============================================================================
# Gemini Message Conversion
# ============================================================================

def convert_messages_to_genai(messages: List[ChatMessage]) -> tuple[Optional[str], List[Content]]:
    """
    Convert OpenAI format messages to Google GenAI format
    Supports multi-turn conversations with tool calling
    
    Important: Gemini requires all function_response to be in the same Content
    """
    system_instruction = None
    contents = []
    
    # Preprocessing: Collect consecutive tool message indices
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.role
        content = msg.content
        
        # System message
        if role == "system":
            if isinstance(content, str):
                system_instruction = content
            i += 1
            continue
        
        # Tool result messages - need to merge consecutive tool messages
        if role == "tool":
            # Collect all consecutive tool messages
            tool_parts = []
            while i < len(messages) and messages[i].role == "tool":
                tool_msg = messages[i]
                tool_call_id = tool_msg.tool_call_id or "unknown"
                tool_content = tool_msg.content
                
                # Attempt to parse JSON
                try:
                    result_data = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
                except (json.JSONDecodeError, TypeError):
                    result_data = {"result": tool_content}
                
                tool_parts.append(Part.from_function_response(
                    name=tool_msg.name or tool_call_id,
                    response=result_data
                ))
                i += 1
            
            # Put all tool responses in the same Content
            if tool_parts:
                contents.append(Content(role="user", parts=tool_parts))
            continue
        
        # Assistant with tool_calls
        if role == "assistant" and msg.tool_calls:
            parts = []
            
            # Add text content first (if any)
            if content:
                if isinstance(content, str):
                    parts.append(Part.from_text(text=content))
            
            # Add function calls
            for idx, tool_call in enumerate(msg.tool_calls):
                try:
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                
                # ⭐ Get thought_signature from server cache (only on the first tool_call)
                if idx == 0:
                    from thought_signature_cache import get_thought_signature
                    thought_signature_bytes = get_thought_signature(tool_call.id)
                    
                    if thought_signature_bytes:
                        from google.genai.types import FunctionCall as GeminiFunctionCall
                        fc_part = Part(
                            function_call=GeminiFunctionCall(
                                name=tool_call.function.name,
                                args=args
                            ),
                            thought_signature=thought_signature_bytes
                        )
                        parts.append(fc_part)
                    else:
                        parts.append(Part.from_function_call(
                            name=tool_call.function.name,
                            args=args
                        ))
                else:
                    parts.append(Part.from_function_call(
                        name=tool_call.function.name,
                        args=args
                    ))
            
            contents.append(Content(role="model", parts=parts))
            i += 1
            continue
        
        # Regular user/assistant messages
        genai_role = "user" if role == "user" else "model"
        
        if isinstance(content, str):
            parts = [Part.from_text(text=content)]
        elif isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(Part.from_text(text=item.get("text", "")))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        header, data = image_url.split(",", 1)
                        mime_type = header.split(";")[0].split(":")[1]
                        parts.append(Part.from_bytes(data=base64.b64decode(data), mime_type=mime_type))
                    else:
                        parts.append(Part.from_uri(file_uri=image_url, mime_type="image/jpeg"))
        else:
            parts = [Part.from_text(text=str(content) if content else "")]
        
        if parts:
            contents.append(Content(role=genai_role, parts=parts))
        
        i += 1
    
    return system_instruction, contents


# ============================================================================
# Claude Message Conversion
# ============================================================================

def convert_messages_to_claude(messages: List[ChatMessage]) -> tuple[Optional[str], List[dict]]:
    """
    Convert OpenAI format messages to Claude format
    Supports multi-turn conversations with tool calling
    """
    system_prompt = None
    claude_messages = []
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        # System message
        if role == "system":
            if isinstance(content, str):
                system_prompt = content
            continue
        
        # Tool result message
        if role == "tool":
            # Claude uses tool_result type
            claude_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": content if isinstance(content, str) else json.dumps(content)
                }]
            })
            continue
        
        # Claude only supports user and assistant roles
        claude_role = "user" if role == "user" else "assistant"
        
        # Assistant with tool_calls
        if role == "assistant" and msg.tool_calls:
            claude_content = []
            # Add text content first (if any)
            if content:
                if isinstance(content, str):
                    claude_content.append({"type": "text", "text": content})
            # Add tool_use blocks
            for tool_call in msg.tool_calls:
                try:
                    input_data = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    input_data = {}
                claude_content.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": input_data
                })
            claude_messages.append({"role": "assistant", "content": claude_content})
            continue
        
        # Regular message processing
        if isinstance(content, str):
            claude_messages.append({"role": claude_role, "content": content})
        elif isinstance(content, list):
            # Handle multi-modal content
            claude_content = []
            for item in content:
                if item.get("type") == "text":
                    claude_content.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        # Base64 image
                        header, data = image_url.split(",", 1)
                        mime_type = header.split(";")[0].split(":")[1]
                        claude_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": data,
                            }
                        })
                    else:
                        # URL image
                        claude_content.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": image_url,
                            }
                        })
            claude_messages.append({"role": claude_role, "content": claude_content})
        else:
            claude_messages.append({"role": claude_role, "content": str(content) if content else ""})
    
    return system_prompt, claude_messages
