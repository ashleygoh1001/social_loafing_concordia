"""Small wrapper around the Gemini API with retries."""

from __future__ import annotations

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
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {response!r}")

        texts: list[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                if part is None:
                    continue
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    texts.append(part_text.strip())

        if texts:
            return "\n".join(texts)

        # Check if we hit MAX_TOKENS with no output (thinking ate the budget)
        for cand in (candidates or []):
            finish_reason = str(getattr(cand, "finish_reason", ""))
            if "MAX_TOKENS" in finish_reason:
                raise RuntimeError(
                    f"Gemini hit MAX_TOKENS with no output — "
                    f"thinking likely consumed entire token budget. "
                    f"Response: {response!r}"
                )

        raise RuntimeError(f"Gemini returned no usable text: {response!r}")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences Gemini sometimes wraps responses in."""
        text = text.strip()
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()

    @staticmethod
    def _sanitize_action_spec(text: str) -> str:
        """
        Repair the call_to_action value in a Concordia action spec JSON string.

        Gemini sometimes produces invalid JSON because the call_to_action
        narrative contains unescaped double quotes, already-escaped quotes
        mixed with unescaped ones, or literal newlines/control characters.
        All of these break json.loads.

        Strategy:
          1. Locate the call_to_action value by finding the known prefix and
             using rfind for the known suffix '", "output_type"' as the boundary.
          2. Unescape any already-escaped quotes so we have a uniform plain string.
          3. Strip control characters (newlines, tabs) that are illegal in JSON strings.
          4. Re-escape everything cleanly from scratch.
        """
        prefix = '"call_to_action": "'
        start_idx = text.find(prefix)
        if start_idx == -1:
            return text
        value_start = start_idx + len(prefix)

        boundary = '", "output_type"'
        end_idx = text.rfind(boundary)
        if end_idx == -1 or end_idx <= value_start:
            return text

        raw_value = text[value_start:end_idx]

        # Unescape any already-escaped quotes for a uniform baseline
        plain = raw_value.replace('\\"', '"')
        # Strip control characters illegal in JSON strings
        plain = plain.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Re-escape backslashes then double quotes, cleanly
        clean = plain.replace('\\', '\\\\').replace('"', '\\"')

        return text[:value_start] + clean + text[end_idx:]

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """
        Concordia expects sample_text to return a bare JSON object when
        asked for action specs. Gemini sometimes prepends narrative prose.
        If the text contains an embedded JSON object, extract just that.
        Also repairs unescaped double quotes inside call_to_action values.
        """
        import json

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return text

        candidate = match.group(0).strip()

        # Fast path: already valid JSON
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

        # Repair path: fix unescaped quotes in call_to_action and retry
        try:
            repaired = SafeGeminiModel._sanitize_action_spec(candidate)
            json.loads(repaired)
            return repaired
        except Exception:
            pass

        # Last resort: return original and let Concordia handle/report it
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
        del terminators, timeout, seed, top_k, kwargs

        temp = self.temperature if temperature is None else temperature
        # Always request enough tokens so thinking doesn't crowd out output.
        # Concordia action specs are small JSON blobs; 2048 is plenty of headroom.
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

        raise RuntimeError(f"Gemini text generation failed: {last_error}")

    def sample_choice(
        self,
        prompt: str,
        responses: tuple[str, ...],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict]:
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
                            collected = [getattr(p, "text", None) for p in parts if p]
                            collected = [t for t in collected if t]
                            if collected:
                                text = "\n".join(collected).strip()
                                break

                m = re.search(r"\d+", text)
                if not m:
                    raise ValueError(f"Could not parse choice from: {text!r}")

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