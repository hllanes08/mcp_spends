"""
MCP Client that uses Gemini as the LLM intermediary.

Flow:
  User prompt → Gemini → decides tool calls → MCP Server executes → Gemini summarizes

Usage:
  python client.py "greet John and add 3 + 5"
"""

import asyncio
import json
import sys
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")


def mcp_tools_to_gemini_tools(mcp_tools: list) -> list[genai_types.Tool]:
    """Convert MCP tool definitions to Gemini function declarations."""
    function_declarations = []
    for tool in mcp_tools:
        # Build properties from MCP's inputSchema
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


async def run(user_prompt: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY in .env or environment.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Discover tools from MCP server
            tools_result = await session.list_tools()
            mcp_tools = tools_result.tools
            print(f"Connected to MCP server. Available tools: {[t.name for t in mcp_tools]}")

            # 2. Convert MCP tools to Gemini format
            gemini_tools = mcp_tools_to_gemini_tools(mcp_tools)

            # 3. Send user prompt to Gemini with tool definitions
            messages = [
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=user_prompt)],
                )
            ]

            system_instruction = (
                "You are a helpful assistant with access to tools.\n"
                "When the user asks about spends or expenses for a specific month, "
                "use the get_spends_by_month tool with the month as an integer "
                "(1 = January, 2 = February, ..., 12 = December).\n"
                "If the user is not logged in yet, call the login tool first "
                "before making any data requests."
            )

            print(f"\nUser: {user_prompt}")
            print("Sending to Gemini...")

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=messages,
                    config=genai_types.GenerateContentConfig(
                        tools=gemini_tools,
                        system_instruction=system_instruction,
                    ),
                )
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                return

            # 4. Process tool calls in a loop until Gemini is done
            while response.candidates[0].content.parts:
                # Collect all function calls from the response
                function_calls = [
                    part for part in response.candidates[0].content.parts
                    if part.function_call
                ]

                if not function_calls:
                    # No more tool calls — Gemini has a final text answer
                    break

                # Add Gemini's response (with function calls) to messages
                messages.append(response.candidates[0].content)

                # Execute each tool call against the MCP server
                function_responses = []
                for part in function_calls:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    print(f"  -> Calling tool: {fc.name}({args})")

                    result = await session.call_tool(fc.name, arguments=args)

                    # Extract text content from MCP result
                    result_text = " ".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                    print(f"  <- Result: {result_text}")

                    function_responses.append(
                        genai_types.Part.from_function_response(
                            name=fc.name,
                            response={"result": result_text},
                        )
                    )

                # Send tool results back to Gemini
                messages.append(
                    genai_types.Content(
                        role="user",
                        parts=function_responses,
                    )
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=messages,
                    config=genai_types.GenerateContentConfig(
                        tools=gemini_tools,
                        system_instruction=system_instruction,
                    ),
                )

            # 5. Print Gemini's final answer
            final_text = response.text
            print(f"\nGemini: {final_text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py \"your prompt here\"")
        sys.exit(1)

    asyncio.run(run(sys.argv[1]))
