"""Real, executable tools for end-to-end tool-call tests (scripts/chat_repl.py /auto).

The model only WRITES a call: <tool_call>{"name": ..., "arguments": {...}}</tool_call>. This module checks that call
against the tool schema (known name, required arguments present, no unknown arguments) and runs the real function, so
the answer the model gives afterwards can be checked against real output.

  calculator        evaluates arithmetic safely (AST walk: numbers and + - * / // % ** only, no names or calls)
  get_current_time  the real current time in an IANA timezone (zoneinfo)
  get_weather       live weather from Open-Meteo (free, no API key; geocoding + current conditions)
"""
import ast
import json
import operator
import re
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression, e.g. '(17 * 23) + 4'.",
        "parameters": {"type": "dict", "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression with numbers and + - * / ( )"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time", "description": "Get the current date and time in a timezone.",
        "parameters": {"type": "dict", "properties": {
            "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Kolkata or Europe/London"}},
            "required": ["timezone"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "dict", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "description": "celsius or fahrenheit"}}, "required": ["city"]}}},
]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")


def calculator(expression):
    return {"expression": expression, "result": _eval(ast.parse(expression.replace("^", "**"), mode="eval"))}


def get_current_time(timezone):
    now = datetime.now(ZoneInfo(timezone))
    return {"timezone": timezone, "datetime": now.strftime("%Y-%m-%d %H:%M"), "weekday": now.strftime("%A")}


_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog", 51: "light drizzle",
        53: "drizzle", 55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
        75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "violent rain showers", 95: "thunderstorm"}


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "tool-test"}), timeout=10) as r:
        return json.load(r)


def get_weather(city, unit="celsius"):
    g = _get("https://geocoding-api.open-meteo.com/v1/search?count=1&name=" + urllib.parse.quote(city))
    if not g.get("results"):
        return {"error": f"city not found: {city}"}
    p = g["results"][0]
    u = "fahrenheit" if str(unit).lower().startswith("f") else "celsius"
    w = _get(f"https://api.open-meteo.com/v1/forecast?latitude={p['latitude']}&longitude={p['longitude']}"
             f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&temperature_unit={u}")["current"]
    return {"city": p["name"], "country": p.get("country"), "temperature": w["temperature_2m"], "unit": u,
            "humidity_percent": w["relative_humidity_2m"], "condition": _WMO.get(w["weather_code"], str(w["weather_code"])),
            "wind_kmh": w["wind_speed_10m"]}


FUNCS = {"calculator": calculator, "get_current_time": get_current_time, "get_weather": get_weather}


def parse_calls(text):
    """All <tool_call> blocks -> list of (call dict or None, raw text, parse error)."""
    out = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            obj = json.loads(m.group(1))
            out.append((obj, m.group(1), None))
        except Exception as e:
            out.append((None, m.group(1), f"invalid JSON: {e}"))
    return out


def check_and_run(call, tools):
    """Validate a parsed call against the tool schemas, run it if it is one of FUNCS. Returns (verdict, result dict)."""
    schemas = {t.get("function", t)["name"]: t.get("function", t) for t in tools}
    name, args = call.get("name"), call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return "FAIL: arguments are not valid JSON", {"error": "bad arguments"}
    if name not in schemas:
        return f"FAIL: unknown tool {name!r}", {"error": f"unknown tool {name}"}
    props = schemas[name].get("parameters", {}).get("properties", {})
    required = schemas[name].get("parameters", {}).get("required", [])
    missing = [r for r in required if r not in args]
    extra = [k for k in args if k not in props]
    if missing or extra:
        return f"FAIL: missing {missing} unknown {extra}", {"error": f"missing {missing}, unknown {extra}"}
    if name not in FUNCS:
        return "OK (schema) but no real function for this tool", {"error": "tool not implemented in this harness"}
    try:
        return "OK: schema valid, executed", FUNCS[name](**args)
    except Exception as e:
        return f"OK (schema) but execution failed: {e}", {"error": str(e)}
