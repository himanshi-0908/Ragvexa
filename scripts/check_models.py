#!/usr/bin/env python
"""
check_models.py — Verify that the NVIDIA key in .env can actually invoke the
configured models.

Usage:
    python scripts/check_models.py            # test LLM_MODEL, JUDGE_MODEL, EVAL_GENERATOR_MODELS
    python scripts/check_models.py --list     # also list every model the key can see
    python scripts/check_models.py --probe    # also probe the known candidate list for working models
    NVIDIA_API_KEY=nvapi-... python scripts/check_models.py   # test a different key

NVIDIA retires models (HTTP 410) and not every key is entitled to every model
(HTTP 404 "Function not found for account"). This prints one line per model
with the HTTP status so you can pick ids that work. The key is never printed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import dotenv_values  # noqa: E402

from llm.model_registry import CANDIDATE_MODELS, probe_models, validate_key  # noqa: E402


def _env():
    values = dict(dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env")))
    for k in ("NVIDIA_API_KEY", "LLM_MODEL", "JUDGE_MODEL", "EVAL_GENERATOR_MODELS"):
        if os.environ.get(k):
            values[k] = os.environ[k]
    return values


def _fmt(res):
    if res.ok:
        note = "" if res.output_tokens else " (no visible text at this token limit; reasoning model, still usable)"
        return f"OK   {res.latency_ms:5d} ms  out_tokens={res.output_tokens}{note}"
    return f"FAIL {res.status} {res.note}".strip()


def main():
    env = _env()
    key = (env.get("NVIDIA_API_KEY") or "").strip(" \"'")
    ok, msg, visible = validate_key(key)
    print(f"key: {'present' if key else 'missing'} ({len(key)} chars) -> {msg}")
    if not ok:
        return 2
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
    results = probe_models(key, [m for _, m in configured], use_cache=False)
    all_ok = True
    for var, model in configured:
        res = results[model]
        all_ok &= res.ok
        print(f"  {var:22s} {model:45s} {_fmt(res)}")

    if "--probe" in sys.argv or not all_ok:
        print("\nProbing candidates:")
        results = probe_models(key, CANDIDATE_MODELS, use_cache=False)
        working = [m for m in CANDIDATE_MODELS if results[m].ok]
        for m in CANDIDATE_MODELS:
            print(f"  {m:45s} {_fmt(results[m])}")
        if working:
            print("\nWorking models for this key:", ", ".join(working))
            print("Set LLM_MODEL / JUDGE_MODEL / EVAL_GENERATOR_MODELS in .env to these ids "
                  "(the app also switches to them automatically when a configured model fails).")
        else:
            print("\nNo candidate worked. Run with --list and try other ids, or check the key on build.nvidia.com.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
