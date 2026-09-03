"""
Flask web app that provides a login UI and a chat interface
powered by Gemini + MCP tools (reuses the logic from client.py).

Usage:
  python app.py
"""

import asyncio
import json
import os
import sys
import secrets
from datetime import date
from contextlib import asynccontextmanager
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from google import genai
from google.genai import types as genai_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
API_BASE_URL = os.getenv("API_BASE_URL", "")

# ---------------------------------------------------------------------------
# Helpers copied from client.py
# ---------------------------------------------------------------------------

def mcp_tools_to_gemini_tools(mcp_tools: list) -> list[genai_types.Tool]:
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
    if len(messages) <= max_items:
        return messages
    return messages[-max_items:]


SYSTEM_INSTRUCTION = (
    "You are a helpful financial assistant with access to tools.\n"
    "Today's date is {today}.\n"
    "Session status: LOGGED IN.\n"
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
    "When the user asks for a yearly summary, which category spent the most in a year, "
    "or an annual breakdown, use the summarize_spends_by_year tool.\n"
    "When the user wants to search or find spends by description or keyword in a specific "
    "month, use the search_spends_by_description_month tool.\n"
    "When the user wants to search or find spends by description or keyword in a specific "
    "year, use the search_spends_by_description_year tool.\n"
    "Be concise in your answers. Present data in tables or bullet points."
)

# ---------------------------------------------------------------------------
# MCP + Gemini processing (async)
# ---------------------------------------------------------------------------

async def _process_prompt(user_message: str, history: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """
    Send *user_message* through Gemini + MCP, return (answer, tool_log, updated_history).
    *history* is a list of {"role": ..., "text": ...} dicts persisted in the Flask session.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not configured.", [], history

    gemini_client = genai.Client(api_key=api_key)

    @asynccontextmanager
    async def mcp_connection():
        if MCP_SERVER_URL:
            async with sse_client(MCP_SERVER_URL) as (read, write):
                async with ClientSession(read, write) as sess:
                    yield sess
        else:
            env = {**os.environ, "MCP_TRANSPORT": "stdio"}
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[SERVER_SCRIPT],
                env=env,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as sess:
                    yield sess

    tool_log: list[dict] = []

    async with mcp_connection() as mcp_session:
        await mcp_session.initialize()

        mcp_tools = (await mcp_session.list_tools()).tools
        gemini_tools = mcp_tools_to_gemini_tools(mcp_tools)

        system_instruction = SYSTEM_INSTRUCTION.format(today=date.today().isoformat())

        # Rebuild Gemini messages from history
        messages = []
        for entry in history:
            role = "user" if entry["role"] == "user" else "model"
            messages.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=entry["text"])],
                )
            )

        # Add current user message
        messages.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=user_message)],
            )
        )
        messages = trim_messages(messages)

        # Call Gemini
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=messages,
                config=genai_types.GenerateContentConfig(
                    tools=gemini_tools,
                    system_instruction=system_instruction,
                ),
            )
        except Exception as e:
            return f"Error calling Gemini: {e}", [], history

        # Tool-call loop
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
                tool_log.append({"tool": fc.name, "args": args})

                result = await mcp_session.call_tool(fc.name, arguments=args)
                result_text = " ".join(
                    c.text for c in result.content if hasattr(c, "text")
                )
                tool_log[-1]["result"] = result_text[:500]

                function_responses.append(
                    genai_types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result_text},
                    )
                )

            messages.append(
                genai_types.Content(role="user", parts=function_responses)
            )

            try:
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=messages,
                    config=genai_types.GenerateContentConfig(
                        tools=gemini_tools,
                        system_instruction=system_instruction,
                    ),
                )
            except Exception as e:
                return f"Error calling Gemini: {e}", tool_log, history

        final_text = response.text if response.text else "(No response)"

        # Update history
        history.append({"role": "user", "text": user_message})
        history.append({"role": "assistant", "text": final_text})
        # Keep history bounded
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]

        return final_text, tool_log, history


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if session.get("logged_in"):
        return redirect(url_for("chat"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()

    if not email or not password:
        return render_template("login.html", error="Ingresa tu correo y contrasena.")

    # Authenticate against the same API the MCP server uses
    import httpx

    try:
        url = f"{API_BASE_URL.rstrip('/')}/api/token/"
        resp = httpx.post(url, json={"username": email, "password": password}, timeout=15)
    except Exception as e:
        return render_template("login.html", error=f"Error de conexion: {e}")

    if resp.status_code == 200:
        data = resp.json()
        token = data.get("access") or data.get("access_token") or data.get("token") or ""
        if not token:
            return render_template("login.html", error="Respuesta inesperada del servidor.")

        # Persist token so MCP server.py can reuse it
        token_file = os.path.join(os.path.dirname(__file__), ".session_token")
        with open(token_file, "w") as f:
            f.write(token)

        session["logged_in"] = True
        session["username"] = email
        session["token"] = token
        session["chat_history"] = []
        return redirect(url_for("chat"))
    else:
        return render_template("login.html", error="Credenciales incorrectas. Intenta de nuevo.")


@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html", username=session.get("username", ""))


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Mensaje vacio."}), 400

    history = session.get("chat_history", [])

    loop = asyncio.new_event_loop()
    try:
        answer, tool_log, updated_history = loop.run_until_complete(
            _process_prompt(user_message, history)
        )
    finally:
        loop.close()

    session["chat_history"] = updated_history

    return jsonify({
        "answer": answer,
        "tools_used": tool_log,
    })


@app.route("/logout")
def logout():
    # Clear MCP session token file
    token_file = os.path.join(os.path.dirname(__file__), ".session_token")
    try:
        os.remove(token_file)
    except FileNotFoundError:
        pass

    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() in ("1", "true")
    app.run(host="0.0.0.0", port=port, debug=debug)
