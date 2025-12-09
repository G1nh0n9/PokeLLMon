#!/usr/bin/env python3
"""Quick test to understand Responses API structure"""

import json
from openai import OpenAI

# Initialize client
client = OpenAI()

# Define a simple tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Make a request
messages = [{"role": "user", "content": "What's the weather in Seoul?"}]

resp = client.beta.messages.create(
    model="gpt-4o-mini-2024-07-18",
    max_tokens=1024,
    tools=tools,
    messages=messages,
    betas=["interop-2024-12-01"]
)

print("=== Response Structure ===")
print(f"Type: {type(resp)}")
print(f"Dir: {[x for x in dir(resp) if not x.startswith('_')]}")
print()

print("=== resp.content ===")
print(f"Type: {type(resp.content)}")
print(f"Length: {len(resp.content) if hasattr(resp.content, '__len__') else 'N/A'}")
print()

if hasattr(resp.content, '__iter__'):
    for i, block in enumerate(resp.content):
        print(f"Block {i}: Type={type(block)}, Dir={[x for x in dir(block) if not x.startswith('_')]}")
        print(f"  Block.__dict__: {block.__dict__ if hasattr(block, '__dict__') else 'N/A'}")
        if hasattr(block, 'type'):
            print(f"  Block.type: {block.type}")
        if hasattr(block, 'text'):
            print(f"  Block.text: {block.text[:100]}")
        if hasattr(block, 'name'):
            print(f"  Block.name: {block.name}")
        if hasattr(block, 'input'):
            print(f"  Block.input: {block.input}")
