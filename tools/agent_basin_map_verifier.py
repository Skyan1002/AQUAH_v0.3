"""Verify whether a basin map matches the user's intent using a vision model."""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image

CLAUDE_LIMIT_B64 = 5 * 1024 * 1024
CLAUDE_BIN_LIMIT = int(CLAUDE_LIMIT_B64 * 0.75)


def _encode_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _compress_image(data: bytes, *, bin_limit: int) -> tuple[str, bytes]:
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        width, height = im.size
        quality_seq = [85, 75, 65, 55]
        while True:
            for q in quality_seq:
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=q, optimize=True)
                new_data = buf.getvalue()
                if len(new_data) <= bin_limit:
                    return "image/jpeg", new_data
            if width < 450:
                return "image/jpeg", new_data
            width = int(width * 0.85)
            height = int(height * 0.85)
            im = im.resize((width, height), Image.LANCZOS)


def _file_to_b64(path: str, *, provider: str) -> tuple[str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "image/png"

    if provider == "anthropic" and len(data) > CLAUDE_BIN_LIMIT:
        mime, data = _compress_image(data, bin_limit=CLAUDE_BIN_LIMIT)
    return mime, _encode_b64(data)


def _call_openai(model, messages, *, api_key, temperature):
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    return client.chat.completions.create(model=model, messages=messages, temperature=temperature).choices[0].message.content


def _call_claude(model, messages, *, api_key, temperature, max_tokens=512):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    return client.messages.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens).content[0].text


def _call_gemini(model: str, parts, api_key=None, temperature=0.1):
    import google.generativeai as genai
    genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel(model).generate_content(
        parts, generation_config={"temperature": temperature}
    ).text


def _call_deepseek(model, messages, *, api_key, temperature):
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def verify_basin_map_by_image(
    *,
    input_text: str,
    image_path: str,
    model_name: str = "gpt-4o",
    temperature: float = 0.2,
) -> Tuple[str, str]:
    system = (
        "You are a hydrologist verifying if the basin boundary shown in the map matches the user's request.\n"
        "You have one basin map image with the boundary and context basemap.\n"
        "Decide whether the basin likely matches the user's intent.\n"
        "If it seems too narrow or does not cover the requested area, recommend expanding the basin to HUC8.\n"
        "Return STRICT JSON with keys: decision (ok|expand) and reason (brief, 1-2 sentences)."
    )
    user_block = (
        f"User request: {input_text}\n"
        "Based on the basin map, decide if the basin matches the request."
    )

    if model_name.startswith(("anthropic/", "claude-")):
        provider = "anthropic"
    elif model_name.startswith(("gemini", "models/")):
        provider = "gemini"
    elif model_name.startswith("deepseek"):
        provider = "deepseek"
    else:
        provider = "openai"

    if provider in {"openai", "deepseek"}:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": []}]
        mime, b64 = _file_to_b64(image_path, provider=provider)
        messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        messages[1]["content"].append({"type": "text", "text": user_block})
        raw = (_call_openai if provider == "openai" else _call_deepseek)(
            model_name,
            messages,
            api_key=None,
            temperature=temperature,
        )
    elif provider == "anthropic":
        mime, b64 = _file_to_b64(image_path, provider=provider)
        blocks = [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            {"type": "text", "text": user_block},
        ]
        raw = _call_claude(
            model_name,
            [{"role": "user", "content": blocks}],
            api_key=None,
            temperature=temperature,
        )
    else:
        raw = _call_gemini(
            model_name,
            [user_block, Image.open(image_path)],
            api_key=None,
            temperature=temperature,
        )

    decision = "ok"
    reason = "No issues detected."
    try:
        parsed = json.loads(raw)
        decision = parsed.get("decision", decision)
        reason = parsed.get("reason", reason)
    except json.JSONDecodeError:
        cleaned = raw.strip().lstrip("```json").rstrip("```").strip()
        try:
            parsed = json.loads(cleaned)
            decision = parsed.get("decision", decision)
            reason = parsed.get("reason", reason)
        except json.JSONDecodeError:
            reason = raw.strip()[:300]

    return decision, reason
