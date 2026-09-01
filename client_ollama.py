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
from datetime import date

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

load_dotenv()

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")


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
        msg = response.choices[0].message

        # Serialize assistant message as a dict so Ollama can parse it on
        # subsequent requests (raw OpenAI objects break the message history).
        assistant_dict = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }
        messages.append(assistant_dict)

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  -> Calling tool: {name}({args})")

            result = await session.call_tool(name, arguments=args)
            result_text = " ".join(
                c.text for c in result.content if hasattr(c, "text")
            )
            print(f"  <- Result: {result_text[:200]}")

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

    final_text = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": final_text})
    return final_text


async def run():
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
        timeout=120.0,
    )

    @asynccontextmanager
    async def mcp_connection():
        if MCP_SERVER_URL:
            print(f"Connecting to remote MCP server: {MCP_SERVER_URL}")
            async with sse_client(MCP_SERVER_URL) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session
        else:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[SERVER_SCRIPT],
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session

    async with mcp_connection() as session:
            await session.initialize()

            mcp_tools = (await session.list_tools()).tools
            print(f"Connected to MCP server. Available tools: {[t.name for t in mcp_tools]}")

            openai_tools = mcp_tools_to_openai_tools(mcp_tools)

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a helpful assistant with access to tools.\n"
                        f"Today's date is {date.today().isoformat()}.\n"
                        "When the user asks about spends or expenses for a specific month, "
                        "use the get_spends_by_month tool with the month as an integer "
                        "(1 = January, 2 = February, ..., 12 = December).\n"
                        "When the user says 'this month' use the current month based on today's date.\n"
                        "When the user wants to create a new spend, use the create_spend tool.\n"
                        "When the user asks to filter or search spends by category or spend type, "
                        "first call get_spend_types to find the matching type ID, then call "
                        "search_spends_by_category with that type ID.\n"
                        "When the user asks to see spends grouped or broken down by location, "
                        "use the get_spends_grouped_by_location tool.\n"
                        "When the user asks for a summary, overview, or report of a month's spending, "
                        "use the summarize_spends tool. Present the results clearly with totals, "
                        "category breakdowns, location breakdowns, and top expenses.\n"
                        "Before making any data requests, call check_session first. "
                        "If the session is active, proceed without logging in again. "
                        "Only call the login tool if check_session says there is no active session.\n"
                        "The user can say 'logout' to clear the saved session."
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
