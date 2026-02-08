"""Simple Gemini agent (chat + CLI).

Requires `google-generativeai` and an API key in GEMINI_API_KEY or GOOGLE_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - handled at runtime
    genai = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - handled at runtime
    load_dotenv = None


def _get_api_key(cli_key: Optional[str]) -> str:
    if cli_key:
        return cli_key
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def _build_model(model_name: str, system: Optional[str], temperature: float, max_tokens: int):
    if genai is None:
        raise RuntimeError(
            "Missing dependency: google-generativeai. Install with: pip install google-generativeai"
        )

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if system:
        return genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system,
            generation_config=generation_config,
        )
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemini agent (CLI + REPL)")
    p.add_argument("--prompt", "-p", help="Single prompt to run; omit to start REPL")
    p.add_argument(
        "--model",
        "-m",
        default="gemini-2.5-pro",
        help="Gemini model name (default: gemini-2.5-pro)",
    )
    p.add_argument("--system", help="Optional system instruction")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--api-key", help="API key (or set GEMINI_API_KEY/GOOGLE_API_KEY)")
    p.add_argument(
        "--fallback-models",
        default="gemini-1.0-pro,gemini-pro",
        help="Comma-separated fallback models to try on 404 errors",
    )
    return p.parse_args()


def _is_model_not_found_error(exc: Exception) -> bool:
    msg = str(exc)
    return "not found for API version" in msg and "models/" in msg


def _send_with_fallback(
    prompt: str,
    model_name: str,
    fallback_models: list[str],
    system: Optional[str],
    temperature: float,
    max_tokens: int,
):
    model = _build_model(model_name, system, temperature, max_tokens)
    chat = model.start_chat(history=[])
    try:
        return chat, chat.send_message(prompt)
    except Exception as exc:
        if not _is_model_not_found_error(exc):
            raise
        for fallback in fallback_models:
            model = _build_model(fallback, system, temperature, max_tokens)
            chat = model.start_chat(history=[])
            try:
                return chat, chat.send_message(prompt)
            except Exception:
                continue
        raise


def run_single_prompt(
    prompt: str,
    model_name: str,
    fallback_models: list[str],
    system: Optional[str],
    temperature: float,
    max_tokens: int,
) -> None:
    _, response = _send_with_fallback(
        prompt,
        model_name,
        fallback_models,
        system,
        temperature,
        max_tokens,
    )
    print(response.text or "")


def run_repl(
    model_name: str,
    fallback_models: list[str],
    system: Optional[str],
    temperature: float,
    max_tokens: int,
) -> None:
    model = _build_model(model_name, system, temperature, max_tokens)
    chat = model.start_chat(history=[])
    print("Gemini REPL. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("\n> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        try:
            response = chat.send_message(prompt)
            print(response.text or "")
        except Exception as exc:
            if not _is_model_not_found_error(exc):
                raise
            chat, response = _send_with_fallback(
                prompt,
                model_name,
                fallback_models,
                system,
                temperature,
                max_tokens,
            )
            print(response.text or "")


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    args = parse_args()
    api_key = _get_api_key(args.api_key)
    if not api_key:
        raise SystemExit("Missing API key. Set GEMINI_API_KEY or pass --api-key.")

    genai.configure(api_key=api_key)
    fallback_models = [m.strip() for m in args.fallback_models.split(",") if m.strip()]

    if args.prompt:
        run_single_prompt(
            args.prompt,
            args.model,
            fallback_models,
            args.system,
            args.temperature,
            args.max_tokens,
        )
        return
    run_repl(
        args.model,
        fallback_models,
        args.system,
        args.temperature,
        args.max_tokens,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Unhandled error: {exc}", file=sys.stderr)
        sys.exit(1)
