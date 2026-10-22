#!/usr/bin/env python
"""
check_models.py — Verify that the NVIDIA key in .env can actually invoke the
configured models.

Usage:
    python scripts/check_models.py            # test LLM_MODEL, JUDGE_MODEL, EVAL_GENERATOR_MODELS
    python scripts/check_models.py --list     # also list every chat model the key can see
    python scripts/check_models.py --probe    # also try to find working models among those listed

NVIDIA retires models (HTTP 410) and not every key is entitled to every model
(HTTP 404 "Function not found for account"). This script prints one line per
model with the HTTP status so you can pick ids that work, then set them in
.env. The key is never printed.
"""

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import dotenv_values  # noqa: E402

BASE = "https://integrate.api.nvidia.com/v1"
PROBE_CANDIDATES = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-nano-3-30b-a3b",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
]


def _env():
    values = dict(dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env")))
    values.update({k: v for k, v in os.environ.items() if k in values or k.startswith(("NVIDIA_", "LLM_", "JUDGE_", "EVAL_"))})
    return values


def _call(key, path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
            msg = detail.get("detail") or detail.get("title") or (detail.get("error") or {}).get("message") or ""
        except Exception:
            msg = ""
        return e.code, str(msg)[:120]
    except Exception as e:  # network
        return "ERR", str(e)[:120]


def test_model(key, model):
    status, info = _call(key, "/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "Reply with the single word OK."}], "max_tokens": 200,
    })
    if status == 200:
        usage = info.get("usage") or {}
        text = ((info.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        note = "" if text.strip() else " (reasoning model: no visible text at this token limit, still usable)"
        return True, f"OK   tokens in/out={usage.get('prompt_tokens')}/{usage.get('completion_tokens')} reply={text.strip()[:30]!r}{note}"
    hint = {410: "retired by NVIDIA", 404: "not enabled for this key", 403: "authorization failed", 401: "bad key"}.get(status, "")
    return False, f"FAIL {status} {hint} {info}".strip()


def main():
    env = _env()
    key = (env.get("NVIDIA_API_KEY") or "").strip(" \"'")
    if not key:
        print("NVIDIA_API_KEY is not set in .env")
        return 2
    print(f"key: present ({len(key)} chars, starts with {key[:6]}…)")

    status, data = _call(key, "/models")
    if status != 200:
        print(f"GET /models failed: {status} {data}")
        return 2
    visible = sorted(m["id"] for m in data.get("data", []))
    print(f"models visible to this key: {len(visible)}")
    if "--list" in sys.argv:
        for m in visible:
            print("  ", m)

    configured = []
    for var in ("LLM_MODEL", "JUDGE_MODEL"):
        if env.get(var):
            configured.append((var, env[var].strip()))
    for m in (env.get("EVAL_GENERATOR_MODELS") or "").split(","):
        if m.strip():
            configured.append(("EVAL_GENERATOR_MODELS", m.strip()))

    print("\nConfigured models:")
    all_ok = True
    for var, model in configured:
        ok, msg = test_model(key, model)
        all_ok &= ok
        print(f"  {var:22s} {model:45s} {msg}")

    if "--probe" in sys.argv or not all_ok:
        print("\nProbing candidates:")
        working = []
        for model in PROBE_CANDIDATES:
            ok, msg = test_model(key, model)
            print(f"  {model:45s} {msg}")
            if ok:
                working.append(model)
        if working:
            print("\nWorking models for this key:", ", ".join(working))
            print("Set LLM_MODEL / JUDGE_MODEL / EVAL_GENERATOR_MODELS in .env to these ids.")
        else:
            print("\nNo candidate worked. Run with --list and try other ids, or check the key on build.nvidia.com.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
