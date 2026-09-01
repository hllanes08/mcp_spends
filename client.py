"""
MCP Client chatbot that uses Gemini as the LLM intermediary.

Flow:
  User prompt → Gemini → decides tool calls → MCP Server executes → Gemini summarizes

Usage:
  python client.py
"""

import asyncio
import json
import sys
import os
from datetime import date

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

load_dotenv()

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))


def mcp_tools_to_gemini_tools(mcp_tools: list) -> list[genai_types.Tool]:
    """Convert MCP tool definitions to Gemini function declarations."""
    function_declarations = []
    for tool in mcp_tools:
        properties = {}
        required = tool.inputSchema.get("required", [])
        for prop_name, prop_info in tool.inputSchema.get("properties", {}).items():
            properties[prop_name] = genai_types.Schema(
                type=_map_json_type(prop_info.get("type", "string")),
                description=prop_info.get("description", ""),
            )

        function_declarations.append(
            genai_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties=properties,
                    required=required,
                ),
            )
        )

    return [genai_types.Tool(function_declarations=function_declarations)]


def _map_json_type(json_type: str) -> str:
    """Map JSON Schema types to Gemini schema types."""
    mapping = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }
    return mapping.get(json_type, "STRING")


def trim_messages(messages: list, max_items: int = MAX_HISTORY * 2) -> list:
    """Keep messages list from growing unbounded."""
    if len(messages) <= max_items:
        return messages
    return messages[-max_items:]


async def process_response(gemini_client, session, messages, gemini_tools, system_instruction, model):
    """Process tool calls in a loop until Gemini gives a final text answer."""
    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=messages,
            config=genai_types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=system_instruction,
            ),
        )
    except Exception as e:
        print(f"\n[Error calling Gemini: {e}]")
        return "(Error communicating with Gemini)"

    while response.candidates and response.candidates[0].content.parts:
        function_calls = [
            part for part in response.candidates[0].content.parts
            if part.function_call
        ]

        if not function_calls:
            break

        messages.append(response.candidates[0].content)

        function_responses = []
        for part in function_calls:
            fc = part.function_call
            args = dict(fc.args) if fc.args else {}
            print(f"  -> Calling tool: {fc.name}({args})")

            result = await session.call_tool(fc.name, arguments=args)
            result_text = " ".join(
                c.text for c in result.content if hasattr(c, "text")
            )
            print(f"  <- Result: {result_text[:200]}")

            function_responses.append(
                genai_types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_text},
                )
            )

        messages.append(
            genai_types.Content(
                role="user",
                parts=function_responses,
            )
        )

        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=messages,
                config=genai_types.GenerateContentConfig(
                    tools=gemini_tools,
                    system_instruction=system_instruction,
                ),
            )
        except Exception as e:
            print(f"\n[Error calling Gemini: {e}]")
            return "(Error communicating with Gemini)"

    final_text = response.text if response.text else "(No response — try rephrasing your question)"
    messages.append(
        genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=final_text)],
        )
    )
    print(f"\nAssistant: {final_text}\n")
    return final_text


async def run():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY in .env or environment.")
        sys.exit(1)

    gemini_client = genai.Client(api_key=api_key)

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

        gemini_tools = mcp_tools_to_gemini_tools(mcp_tools)

        # Check session status once at startup
        session_result = await session.call_tool("check_session", arguments={})
        session_status = " ".join(
            c.text for c in session_result.content if hasattr(c, "text")
        )
        is_logged_in = "active" in session_status.lower()

        system_instruction = (
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
        )

        messages = []

        print(f"\nChatbot ready! Using model: {GEMINI_MODEL}")
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

            messages.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=user_input)],
                )
            )
            messages = trim_messages(messages)

            await process_response(
                gemini_client, session, messages, gemini_tools,
                system_instruction, GEMINI_MODEL,
            )


if __name__ == "__main__":
    asyncio.run(run())
