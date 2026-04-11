# retrieval/requirement_extractor.py
# Calls the LLM to extract structured requirements from a
# natural language query. Output feeds the catalog filter.

import json
import os
import requests
import time
from typing import Dict, Any

GEMINI_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-2.0-flash:generateContent"
)
API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """\
You are a requirements parser for embedded systems.
Given a user query about microcontroller needs, extract
structured requirements as JSON.

Return ONLY valid JSON with these fields (omit fields
that are not mentioned or cannot be inferred):

{
  "connectivity"    : ["BLE", "WiFi", "Ethernet", ...],
  "max_power_ma"    : <float or null>,
  "min_flash_kb"    : <int or null>,
  "min_ram_kb"      : <int or null>,
  "min_freq_mhz"    : <int or null>,
  "use_case"        : "ultra-low-power" | "wireless-ble"
                    | "high-performance" | "general-purpose"
                    | null,
  "peripherals"     : ["ADC", "DAC", "CAN", ...]
}

Rules:
- Only include a field if the user clearly needs it.
- connectivity values must match exactly:
  BLE, WiFi, Ethernet, USB, UART, SPI, I2C, CAN, 802.15.4
- If the user says "low power" without a number,
  set use_case to "ultra-low-power", not max_power_ma.
- Return raw JSON only. No markdown, no explanation.\
"""


def _call_gemini(prompt: str) -> str:
    for attempt in range(3):
        resp = requests.post(
            f"{GEMINI_URL}?key={API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 256
                }
            },
            timeout=30
        )

        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"[requirement_extractor] Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    return "{}"


def extract_requirements(query: str) -> Dict[str, Any]:
    """
    First tries keyword matching for speed and to save API quota.
    Falls back to Gemini only if keywords are insufficient.
    """
    query_lower = query.lower()
    requirements = {}

    # Connectivity keywords
    connectivity = []
    if "ble" in query_lower or "bluetooth" in query_lower:
        connectivity.append("BLE")
    if "ethernet" in query_lower:
        connectivity.append("Ethernet")
    if "wifi" in query_lower or "wi-fi" in query_lower:
        connectivity.append("WiFi")
    if "usb" in query_lower:
        connectivity.append("USB")
    if "can" in query_lower:
        connectivity.append("CAN")
    if connectivity:
        requirements["connectivity"] = connectivity

    # Use case keywords : BLE takes priority over low-power
    if "ble" in query_lower or "bluetooth" in query_lower:
        requirements["use_case"] = "wireless-ble"
    elif any(w in query_lower for w in ["low power", "ultra low", "battery", "wearable", "sleep"]):
        requirements["use_case"] = "ultra-low-power"
    elif any(w in query_lower for w in ["high performance", "fast", "480", "ethernet", "dsp"]):
        requirements["use_case"] = "high-performance"
    elif any(w in query_lower for w in ["general", "hobby", "cheap", "simple"]):
        requirements["use_case"] = "general-purpose"

    # Flash keywords
    if "1mb" in query_lower or "1024kb" in query_lower:
        requirements["min_flash_kb"] = 1024
    if "512" in query_lower:
        requirements["min_flash_kb"] = 512

    # If we got something useful from keywords, skip Gemini entirely
    if requirements:
        print(f"[requirement_extractor] Keyword match: {requirements}")
        return requirements

    # Only call Gemini if keywords found nothing
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser query: {query}"
    try:
        raw = _call_gemini(full_prompt)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"[requirement_extractor] Warning: {e}. Returning empty requirements.")
        return {}