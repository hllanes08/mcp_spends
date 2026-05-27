"""
MCP Client chatbot that uses a local Ollama model as the LLM intermediary.

Flow:
  User prompt → Ollama (Qwen) → decides tool calls → MCP Server executes → Ollama summarizes

Usage:
  python client_ollama.py
"""

import asyncio
import json
import sys
import os

from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")


def mcp_tools_to_openai_tools(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function calling format."""
    tools = []
    for tool in mcp_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })
    return tools


async def process_response(client, session, messages, openai_tools, response):
    """Process tool calls in a loop until the model gives a final text answer."""
    while response.choices[0].message.tool_calls:
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  -> Calling tool: {name}({args})")

            result = await session.call_tool(name, arguments=args)
            result_text = " ".join(
                c.text for c in result.content if hasattr(c, "text")
            )
            print(f"  <- Result: {result_text}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=openai_tools,
        )

    final_text = response.choices[0].message.content
    messages.append({"role": "assistant", "content": final_text})
    return final_text


async def run():
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        timeout=120.0,
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = (await session.list_tools()).tools
            print(f"Connected to MCP server. Available tools: {[t.name for t in mcp_tools]}")

            openai_tools = mcp_tools_to_openai_tools(mcp_tools)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant with access to tools.\n"
                        "When the user asks about spends or expenses for a specific month, "
                        "use the get_spends_by_month tool with the month as an integer "
                        "(1 = January, 2 = February, ..., 12 = December).\n"
                        "When the user wants to create a new spend, use the create_spend tool.\n"
                        "If the user is not logged in yet, call the login tool first "
                        "before making any data requests."
                    ),
                },
            ]

            print(f"\nChatbot ready! Using model: {OLLAMA_MODEL}")
            print("Type 'quit' or 'exit' to stop.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break

                messages.append({"role": "user", "content": user_input})

                response = client.chat.completions.create(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    tools=openai_tools,
                )

                final_text = await process_response(
                    client, session, messages, openai_tools, response
                )
                print(f"\nAssistant: {final_text}\n")


if __name__ == "__main__":
    asyncio.run(run())
