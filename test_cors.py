"""
Test Vertex AI Proxy (Gemini + Claude)
"""
import requests
import json

# Configuration
ENDPOINT_URL = "http://localhost:8080"  # Change to your Cloud Run URL if deployed
API_KEY = "your-api-key-here"  # Set your MASTER_KEY

# Remote endpoint example
# ENDPOINT_URL = "https://your-project.run.app"


def test_health():
    """Test health check"""
    print("=" * 60)
    print("1. Test Health Check (/health)")
    print("=" * 60)
    
    try:
        response = requests.get(f"{ENDPOINT_URL}/health")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
        if response.status_code == 200:
            print("   ✅ Health check passed!")
        else:
            print("   ❌ Health check failed!")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def test_models():
    """Test model listing"""
    print("=" * 60)
    print("2. Test Model Listing (/v1/models)")
    print("=" * 60)
    
    try:
        response = requests.get(
            f"{ENDPOINT_URL}/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            
            gemini_models = [m for m in models if m.get("owned_by") == "google"]
            claude_models = [m for m in models if m.get("owned_by") == "anthropic"]
            
            print(f"\n   Gemini Models ({len(gemini_models)}):")
            for model in gemini_models[:5]:
                print(f"     - {model.get('id')}")
            if len(gemini_models) > 5:
                print(f"     ... and {len(gemini_models) - 5} more")
            
            print(f"\n   Claude Models ({len(claude_models)}):")
            for model in claude_models[:5]:
                print(f"     - {model.get('id')}")
            if len(claude_models) > 5:
                print(f"     ... and {len(claude_models) - 5} more")
            
            print("\n   ✅ Model listing successful!")
        else:
            print(f"   ❌ Request failed: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def test_gemini_chat():
    """Test Gemini chat"""
    print("=" * 60)
    print("3. Test Gemini Chat (Non-streaming)")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-3-flash-preview",
                "messages": [
                    {"role": "user", "content": "Say 'Hello from Gemini!' and nothing else."}
                ],
                "stream": False,
                "max_tokens": 50
            }
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"   Response Content: {content}")
            print("   ✅ Gemini chat successful!")
        else:
            print(f"   ❌ Request failed: {response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def test_claude_chat():
    """Test Claude chat"""
    print("=" * 60)
    print("4. Test Claude Chat (Non-streaming)")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-sonnet-4-5",
                "messages": [
                    {"role": "user", "content": "Say 'Hello from Claude!' and nothing else."}
                ],
                "stream": False,
                "max_tokens": 50
            }
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"   Response Content: {content}")
            print("   ✅ Claude chat successful!")
        else:
            print(f"   ❌ Request failed: {response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def test_gemini_stream():
    """Test Gemini streaming chat"""
    print("=" * 60)
    print("5. Test Gemini Streaming Chat")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-3-flash-preview",
                "messages": [
                    {"role": "user", "content": "Count from 1 to 3."}
                ],
                "stream": True,
                "max_tokens": 50
            },
            stream=True
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   Streaming Response:")
            full_content = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        try:
                            data = json.loads(decoded[6:])
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                full_content += content
                        except:
                            pass
            print(f"     {full_content}")
            print("   ✅ Gemini streaming chat successful!")
        else:
            print(f"   ❌ Request failed: {response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def test_claude_stream():
    """Test Claude streaming chat"""
    print("=" * 60)
    print("6. Test Claude Streaming Chat")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-haiku-3",
                "messages": [
                    {"role": "user", "content": "Count from 1 to 3."}
                ],
                "stream": True,
                "max_tokens": 50
            },
            stream=True
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   Streaming Response:")
            full_content = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        try:
                            data = json.loads(decoded[6:])
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                full_content += content
                        except:
                            pass
            print(f"     {full_content}")
            print("   ✅ Claude streaming chat successful!")
        else:
            print(f"   ❌ Request failed: {response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def test_gemini_tool_call():
    """Test Gemini tool calling"""
    print("=" * 60)
    print("7. Test Gemini Tool Calling")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-3-flash-preview",
                "messages": [
                    {"role": "user", "content": "What is the weather in Tokyo?"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name, e.g. Tokyo"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }],
                "stream": False,
                "max_tokens": 200
            }
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            message = data.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            finish_reason = data.get("choices", [{}])[0].get("finish_reason", "")
            
            if tool_calls:
                print(f"   Tool Calls Count: {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"   - Function Name: {tc.get('function', {}).get('name')}")
                    print(f"     Arguments: {tc.get('function', {}).get('arguments')}")
                print(f"   finish_reason: {finish_reason}")
                print("   ✅ Gemini tool calling successful!")
            else:
                content = message.get("content", "")
                print(f"   Response Content: {content[:100]}...")
                print("   ⚠️ Model did not return tool calls (may have answered the question directly)")
        else:
            print(f"   ❌ Request failed: {response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def test_claude_tool_call():
    """Test Claude tool calling"""
    print("=" * 60)
    print("8. Test Claude Tool Calling")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-haiku-3.5",
                "messages": [
                    {"role": "user", "content": "What is the weather in San Francisco?"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The unit of temperature"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }],
                "stream": False,
                "max_tokens": 200
            }
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            message = data.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            finish_reason = data.get("choices", [{}])[0].get("finish_reason", "")
            
            if tool_calls:
                print(f"   Tool Calls Count: {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"   - Function Name: {tc.get('function', {}).get('name')}")
                    print(f"     Arguments: {tc.get('function', {}).get('arguments')}")
                print(f"   finish_reason: {finish_reason}")
                print("   ✅ Claude tool calling successful!")
            else:
                content = message.get("content", "")
                print(f"   Response Content: {content[:100]}...")
                print("   ⚠️ Model did not return tool calls (may have answered the question directly)")
        else:
            print(f"   ❌ Request failed: {response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def test_responses_api():
    """Test Responses API (New Standard)"""
    print("=" * 60)
    print("9. Test Responses API (/v1/responses)")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/responses",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-3-flash-preview",
                "input": "Tell me a three sentence bedtime story about a unicorn.",
                "max_output_tokens": 200
            }
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response ID: {data.get('id')}")
            print(f"   Status: {data.get('status')}")
            
            output = data.get("output", [])
            if output:
                for item in output:
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for c in content:
                            if c.get("type") == "output_text":
                                text = c.get("text", "")[:100]
                                print(f"   Content: {text}...")
            
            usage = data.get("usage", {})
            print(f"   Token Usage: input={usage.get('input_tokens')}, output={usage.get('output_tokens')}")
            print("   ✅ Responses API test successful!")
        else:
            print(f"   ❌ Request failed: {response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def test_responses_api_with_instructions():
    """Test Responses API (with instructions)"""
    print("=" * 60)
    print("10. Test Responses API (with instructions)")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/responses",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-sonnet-4.5",
                "instructions": "You are a pirate who speaks in pirate dialect.",
                "input": "Introduce yourself in one sentence.",
                "max_output_tokens": 100
            }
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response ID: {data.get('id')}")
            
            output = data.get("output", [])
            if output:
                for item in output:
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for c in content:
                            if c.get("type") == "output_text":
                                print(f"   Content: {c.get('text', '')}")
            print("   ✅ Responses API (instructions) test successful!")
        else:
            print(f"   ❌ Request failed: {response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    print()


def main():
    print("\n" + "=" * 60)
    print("Vertex AI Proxy Test (Gemini + Claude)")
    print(f"Endpoint: {ENDPOINT_URL}")
    print("=" * 60 + "\n")
    
    test_health()
    test_models()
    test_gemini_chat()
    test_claude_chat()
    test_gemini_stream()
    test_claude_stream()
    test_gemini_tool_call()
    test_claude_tool_call()
    test_responses_api()
    test_responses_api_with_instructions()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
