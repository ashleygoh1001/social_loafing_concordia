"""Small wrapper around the Gemini API with retries."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from google import genai
from google.genai import types


class SafeGeminiModel:
    def __init__(
        self,
        *,
        model_name: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        temperature: float = 0.6,
        max_output_tokens: int = 10000,
        max_retries: int = 8,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment (or api_key).")

        self.client = genai.Client(api_key=self.api_key)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

    def _extract_text(self, response: Any) -> str:
        # Fast path: response.text
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {response!r}")

        # Check for MAX_TOKENS before trying to collect parts
        for cand in candidates:
            if "MAX_TOKENS" in str(getattr(cand, "finish_reason", "")):
                raise RuntimeError(
                    "Gemini hit MAX_TOKENS with no output — "
                    "thinking likely consumed entire token budget. "
                    f"Response: {response!r}"
                )

        texts: list[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    texts.append(part_text.strip())

        if texts:
            return "\n".join(texts)

        raise RuntimeError(f"Gemini returned no usable text: {response!r}")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences Gemini sometimes wraps responses in."""
        text = text.strip()
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """
        Concordia expects sample_text to return a bare JSON object for action
        specs. Handles two failure modes from Gemini:
          1. Narrative prose wrapping a JSON object — extract just the object.
          2. The entire response is a JSON-encoded string (i.e. Gemini wrapped
             the JSON object in outer quotes, producing "{\"key\": \"val\"}").
             Decode the outer string layer first, then extract the object.
        """
        text = text.strip()

        # Case 2: outer-quoted JSON string — decode one layer, then fall through
        if text.startswith('"') and text.endswith('"'):
            try:
                inner = json.loads(text)   # produces the raw JSON object string
                if isinstance(inner, str):
                    text = inner.strip()
            except Exception:
                pass

        # Case 1 (and Case 2 after unwrapping): find embedded JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            candidate = match.group(0).strip()
            try:
                json.loads(candidate)  # validate
                return candidate
            except Exception:
                pass

        return text

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        terminators=(),
        temperature: float | None = None,
        timeout=None,
        seed=None,
        top_k=None,
        top_p=None,
        **kwargs: Any,
    ) -> str:
        # Unused by Gemini API
        del terminators, timeout, seed, top_k, kwargs

        temp = self.temperature if temperature is None else temperature
        # Always request enough tokens so thinking doesn't crowd out output.
        out_tokens = max(self.max_output_tokens if max_tokens is None else max_tokens, 2048)

        config: types.GenerateContentConfigDict = {
            "temperature": temp,
            "max_output_tokens": out_tokens,
            # Disable thinking for sample_text — Concordia calls this very
            # frequently for small structured outputs, and thinking burns tokens.
            "thinking_config": {"thinking_budget": 0},
        }
        if top_p is not None:
            config["top_p"] = top_p

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                text = self._extract_text(response)
                text = self._strip_code_fences(text)
                text = self._extract_json_from_text(text)
                return text
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 60)
                    print(f"[sample_text] attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)

        raise RuntimeError(f"Gemini text generation failed after {self.max_retries} attempts: {last_error}")

    def sample_choice(
        self,
        prompt: str,
        responses: tuple[str, ...],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict]:
        del seed  # not used by Gemini API

        forced_prompt = (
            prompt
            + "\n\nChoose exactly one option.\n"
            + "\n".join(f"{i}: {r}" for i, r in enumerate(responses))
            + "\n\nReply with only the index number."
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=forced_prompt,
                    config={
                        "temperature": 0.0,
                        "max_output_tokens": 64,
                        "thinking_config": {"thinking_budget": 0},
                    },
                )

                text = (getattr(response, "text", None) or "").strip()
                if not text:
                    for cand in getattr(response, "candidates", []) or []:
                        content = getattr(cand, "content", None)
                        parts = getattr(content, "parts", None) if content else None
                        if parts:
                            collected = [
                                getattr(p, "text", None)
                                for p in parts if p
                            ]
                            text = "\n".join(t for t in collected if t).strip()
                            if text:
                                break

                m = re.search(r"\d+", text)
                if not m:
                    raise ValueError(f"Could not parse choice index from: {text!r}")

                idx = int(m.group(0))
                if not 0 <= idx < len(responses):
                    idx = 0

                return idx, text, {}

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 60)
                    print(f"[sample_choice] attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)

        print(f"[sample_choice] all retries exhausted, defaulting to index 0. Last error: {last_error}")
        return 0, "", {}