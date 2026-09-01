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
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))


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


def trim_messages(messages: list, max_pairs: int = MAX_HISTORY) -> list:
    """Keep the system prompt + last N user/assistant pairs to limit context."""
    if len(messages) <= 1 + max_pairs * 2:
        return messages
    return [messages[0]] + messages[-(max_pairs * 2):]


import re

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen output."""
    return _THINK_RE.sub("", text).strip()


def stream_response(client, model, messages, tools, show_label=True):
    """Stream a chat completion, printing tokens as they arrive."""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
        )
    except Exception as e:
        print(f"\n[Error calling LLM: {e}]")
        return "", None

    content_parts = []
    tool_calls_by_index: dict[int, dict] = {}
    in_think = False
    label_printed = False

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # Stream text content, hiding <think> blocks
        if delta.content:
            content_parts.append(delta.content)
            # Buffer and detect think blocks
            partial = "".join(content_parts)
            if "<think>" in partial and "</think>" not in partial:
                in_think = True
                continue
            if in_think and "</think>" in partial:
                in_think = False
                visible = _strip_think(partial)
                content_parts.clear()
                content_parts.append(visible)
                if visible and show_label:
                    if not label_printed:
                        print("\nAssistant: ", end="", flush=True)
                        label_printed = True
                    print(visible, end="", flush=True)
                continue
            if not in_think:
                if show_label and not label_printed:
                    print("\nAssistant: ", end="", flush=True)
                    label_printed = True
                if show_label:
                    print(delta.content, end="", flush=True)

        # Accumulate tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_by_index:
                    tool_calls_by_index[idx] = {
                        "id": tc.id or "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = tool_calls_by_index[idx]
                if tc.id:
                    entry["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        entry["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        entry["function"]["arguments"] += tc.function.arguments

    raw_content = "".join(content_parts)
    content = _strip_think(raw_content)
    tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)] if tool_calls_by_index else None

    if not tool_calls and content and label_printed:
        print(flush=True)

    return content, tool_calls


def fallback_response(client, model, messages, tools):
    """Non-streaming fallback when stream returns empty."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=False,
        )
        msg = response.choices[0].message
        content = _strip_think(msg.content or "")
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return content, tool_calls
    except Exception as e:
        print(f"\n[Fallback error: {e}]")
        return "", None


async def process_response(client, session, messages, openai_tools):
    """Process tool calls in a loop until the model gives a final text answer."""
    content, tool_calls = stream_response(client, OLLAMA_MODEL, messages, openai_tools)

    # Retry with non-streaming if stream returned nothing
    if not content and not tool_calls:
        content, tool_calls = fallback_response(client, OLLAMA_MODEL, messages, openai_tools)
        if content:
            print(f"\nAssistant: {content}", flush=True)

    while tool_calls:
        assistant_dict = {
            "role": "assistant",
            "content": content or "",
            "tool_calls": tool_calls,
        }
        messages.append(assistant_dict)

        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            print(f"  -> Calling tool: {name}({args})")

            result = await session.call_tool(name, arguments=args)
            result_text = " ".join(
                c.text for c in result.content if hasattr(c, "text")
            )
            print(f"  <- Result: {result_text[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_text,
            })

        content, tool_calls = stream_response(client, OLLAMA_MODEL, messages, openai_tools, show_label=True)
        if not content and not tool_calls:
            content, tool_calls = fallback_response(client, OLLAMA_MODEL, messages, openai_tools)
            if content:
                print(f"\nAssistant: {content}", flush=True)

    final_text = content or "(No response — try rephrasing your question)"
    messages.append({"role": "assistant", "content": final_text})
    if not content:
        print(f"\nAssistant: {final_text}", flush=True)
    print()
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
            env = {**os.environ, "MCP_TRANSPORT": "stdio"}
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[SERVER_SCRIPT],
                env=env,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session

    async with mcp_connection() as session:
        await session.initialize()

        mcp_tools = (await session.list_tools()).tools
        print(f"Connected to MCP server. Available tools: {[t.name for t in mcp_tools]}")

        openai_tools = mcp_tools_to_openai_tools(mcp_tools)

        # Check session status once at startup to inform the system prompt
        session_result = await session.call_tool("check_session", arguments={})
        session_status = " ".join(
            c.text for c in session_result.content if hasattr(c, "text")
        )
        is_logged_in = "active" in session_status.lower()

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant with access to tools.\n"
                    f"Today's date is {date.today().isoformat()}.\n"
                    f"Session status: {'LOGGED IN — do not call login or check_session unless the user asks to switch accounts.' if is_logged_in else 'NOT LOGGED IN — call login before making data requests.'}\n"
                    "When the user asks about spends or expenses for a specific month, "
                    "use the get_spends_by_month tool with the month as an integer "
                    "(1 = January, 2 = February, ..., 12 = December).\n"
                    "When the user says 'this month' use the current month based on today's date.\n"
                    "When the user asks about spends for an entire year or annual totals, "
                    "use the get_spends_by_year tool with the year number (e.g. 2026).\n"
                    "When the user says 'this year' use the current year based on today's date.\n"
                    "When the user only needs the total amount for a year (not individual spends), "
                    "use get_yearly_total — it's faster.\n"
                    "When the user wants to compare totals across multiple years, "
                    "use get_multi_year_totals with comma-separated years (e.g. '2024,2025,2026').\n"
                    "When the user wants to create a new spend, use the create_spend tool.\n"
                    "When the user asks to filter or search spends by category or spend type, "
                    "first call get_spend_types to find the matching type ID, then call "
                    "search_spends_by_category with that type ID.\n"
                    "When the user asks to see spends grouped or broken down by location, "
                    "use the get_spends_grouped_by_location tool.\n"
                    "When the user asks for a summary, overview, or report of a month's spending, "
                    "use the summarize_spends tool. Present the results clearly with totals, "
                    "category breakdowns, location breakdowns, and top expenses.\n"
                    "The user can say 'logout' to clear the saved session.\n"
                    "Be concise in your answers. Present data in tables or bullet points."
                ),
            },
        ]

        print(f"\nChatbot ready! Using model: {OLLAMA_MODEL}")
        print(f"Session: {'Active' if is_logged_in else 'Not logged in'}")
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
            messages = trim_messages(messages)

            await process_response(client, session, messages, openai_tools)


if __name__ == "__main__":
    asyncio.run(run())
