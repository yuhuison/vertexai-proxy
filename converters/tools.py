"""
Tool Format Conversion - OpenAI Format <-> Gemini/Claude Format
"""
import json
import uuid
from typing import List, Optional

from google.genai import types

from models import Tool, ToolCall, FunctionCall


# ============================================================================
# OpenAI -> Gemini Tool Conversion
# ============================================================================

def convert_tools_to_gemini(tools: List[Tool]) -> types.Tool:
    """
    Convert OpenAI format tools to Gemini format
    
    OpenAI format:
    {"type": "function", "function": {"name": "xxx", "description": "xxx", "parameters": {...}}}
    
    Gemini format:
    types.Tool(function_declarations=[types.FunctionDeclaration(...)])
    
    Note: Must use parameters_json_schema field of types.FunctionDeclaration
    to pass the original JSON Schema, instead of passing dict directly. 
    Passing dict directly will lead to validation by the SDK's Pydantic model 
    according to Schema type, resulting in "Extra inputs are not permitted" error.
    """
    function_declarations = []
    
    for tool in tools:
        if tool.type != "function":
            continue
        
        func = tool.function
        
        decl_kwargs = {"name": func.name}
        
        if func.description:
            decl_kwargs["description"] = func.description
        
        if func.parameters:
            # Use parameters_json_schema to pass original JSON Schema
            # This ensures SDK correctly handles oneOf, anyOf and other JSON Schema features
            decl_kwargs["parameters_json_schema"] = func.parameters
        
        function_declarations.append(types.FunctionDeclaration(**decl_kwargs))
    
    return types.Tool(function_declarations=function_declarations)


def convert_tool_choice_to_gemini(tool_choice: Optional[str | dict]) -> Optional[types.ToolConfig]:
    """Convert OpenAI's tool_choice to Gemini's tool_config"""
    if tool_choice is None:
        return None
    
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            )
        elif tool_choice == "auto":
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )
        elif tool_choice == "required":
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )
    elif isinstance(tool_choice, dict):
        # {"type": "function", "function": {"name": "xxx"}}
        if tool_choice.get("type") == "function":
            func_name = tool_choice.get("function", {}).get("name")
            if func_name:
                return types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[func_name]
                    )
                )
    
    return None


# ============================================================================
# OpenAI -> Claude Tool Conversion
# ============================================================================

def convert_tools_to_claude(tools: List[Tool]) -> List[dict]:
    """
    Convert OpenAI format tools to Claude format
    
    OpenAI format:
    {"type": "function", "function": {"name": "xxx", "description": "xxx", "parameters": {...}}}
    
    Claude format:
    {"name": "xxx", "description": "xxx", "input_schema": {...}}
    """
    claude_tools = []
    
    for tool in tools:
        if tool.type != "function":
            continue
        
        func = tool.function
        claude_tool = {
            "name": func.name,
        }
        
        if func.description:
            claude_tool["description"] = func.description
        
        if func.parameters:
            claude_tool["input_schema"] = func.parameters
        else:
            # Claude needs at least an empty input_schema
            claude_tool["input_schema"] = {"type": "object", "properties": {}}
        
        claude_tools.append(claude_tool)
    
    return claude_tools


def convert_tool_choice_to_claude(tool_choice: Optional[str | dict]) -> Optional[dict]:
    """Convert OpenAI's tool_choice to Claude's tool_choice"""
    if tool_choice is None:
        return None
    
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            # Claude does not have a direct "none", achieved by not passing tools
            return None
        elif tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "required":
            return {"type": "any"}
    elif isinstance(tool_choice, dict):
        # {"type": "function", "function": {"name": "xxx"}}
        if tool_choice.get("type") == "function":
            func_name = tool_choice.get("function", {}).get("name")
            if func_name:
                return {"type": "tool", "name": func_name}
    
    return None


# ============================================================================
# Gemini/Claude Response -> OpenAI Tool Call Conversion
# ============================================================================

def convert_gemini_function_call(function_call, thought_signature: str = None) -> ToolCall:
    """
    Convert Gemini's function_call to OpenAI's tool_call format
    
    Gemini function_call: name, args (dict)
    OpenAI tool_call: id, type, function: {name, arguments (JSON string)}
    
    Note: Gemini 3.0+ requires thought_signature for multi-turn tool calling
    """
    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:24]}",
        type="function",
        function=FunctionCall(
            name=function_call.name,
            arguments=json.dumps(function_call.args) if function_call.args else "{}"
        ),
        thought_signature=thought_signature
    )


def convert_claude_tool_use(tool_use_block) -> ToolCall:
    """
    Convert Claude's tool_use block to OpenAI's tool_call format
    
    Claude tool_use: id, name, input (dict)
    OpenAI tool_call: id, type, function: {name, arguments (JSON string)}
    """
    # tool_use_block can be a dict or an object
    if isinstance(tool_use_block, dict):
        tool_id = tool_use_block.get("id", f"call_{uuid.uuid4().hex[:24]}")
        name = tool_use_block.get("name", "")
        input_data = tool_use_block.get("input", {})
    else:
        tool_id = getattr(tool_use_block, "id", f"call_{uuid.uuid4().hex[:24]}")
        name = getattr(tool_use_block, "name", "")
        input_data = getattr(tool_use_block, "input", {})
    
    return ToolCall(
        id=tool_id,
        type="function",
        function=FunctionCall(
            name=name,
            arguments=json.dumps(input_data) if input_data else "{}"
        )
    )
