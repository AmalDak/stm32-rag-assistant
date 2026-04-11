# generation/llm_client.py
import os
import time
import requests

GEMINI_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-2.0-flash:generateContent"
)
API_KEY = os.environ.get("GEMINI_API_KEY", "")


def generate(prompt: str) -> str:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512
        }
    }

    for attempt in range(3):
        resp = requests.post(
            f"{GEMINI_URL}?key={API_KEY}",
            json=payload,
            timeout=30
        )

        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"[llm_client] Rate limited. Waiting {wait}s before retry {attempt + 1}/3...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    return "Rate limit reached. Please wait a moment and try again."