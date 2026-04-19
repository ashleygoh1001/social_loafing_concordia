"""Run N simulation trials in parallel using Ray.

Parallelism strategy
---------------------
Trial level (Ray): each trial runs as a Ray remote task, so up to
`max_concurrent` trials are in-flight at once across all available CPUs.
A shared EmbedderActor per node loads SentenceTransformer once instead of
once per trial.

The Gemini calls within each trial use the same proven sync google.genai SDK
as safe_gemini_model.py — no aiohttp, no event-loop conflicts with Concordia's
own internal asyncio concurrency.

Usage
-----
    # Local (auto-detects CPUs):
    python run_parallel_ray.py --n_trials 100 --max_steps 20

    # Multi-node (start Ray first, then pass the head address):
    ray start --head --port 6379
    python run_parallel_ray.py --address auto --n_trials 500

    # Tune concurrency to stay under Gemini RPM quota:
    python run_parallel_ray.py --n_trials 100 --max_concurrent 16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import ray
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo / package path setup (mirrors cs_group_project_sim.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_CONCORDIA_PARENT = _REPO_ROOT / "concordia"
if _LOCAL_CONCORDIA_PARENT.exists():
    sys.path.insert(0, str(_LOCAL_CONCORDIA_PARENT))

from cs_group_project_sim import (
    TRAIT_POOL_PATH,
    build_simulation,
    load_trait_pool,
    sample_profiles,
)


# ---------------------------------------------------------------------------
# Sync Gemini model — identical logic to safe_gemini_model.py, self-contained
# so Ray workers don't need to import from the simulations directory.
# ---------------------------------------------------------------------------

class _GeminiModel:
    """Sync Gemini wrapper with exponential-backoff retries."""

    def __init__(
        self,
        *,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.6,
        max_output_tokens: int = 10_000,
        max_retries: int = 8,
    ) -> None:
        from google import genai

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.client = genai.Client(api_key=api_key)
        self._call_times: list[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidates = getattr(response, "candidates", None) or []
        texts: list[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            for part in (parts or []):
                t = getattr(part, "text", None)
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        if texts:
            return "\n".join(texts)
        raise RuntimeError(f"Gemini returned no usable text: {response!r}")

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            candidate = m.group(0).strip()
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass
        return text

    def _call(self, contents: str, config: dict) -> Any:
        from google.genai import types
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                _t0 = time.perf_counter()
                result = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                _elapsed = time.perf_counter() - _t0
                self._call_times.append(_elapsed)
                return result
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 60)
                    print(f"[Gemini] attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        raise RuntimeError(f"Gemini call failed after {self.max_retries} retries: {last_err}")

    # ------------------------------------------------------------------
    # Concordia-compatible public API
    # ------------------------------------------------------------------
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
        **kwargs,
    ) -> str:
        out_tokens = max(
            self.max_output_tokens if max_tokens is None else max_tokens,
            2048,
        )
        config: dict = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_output_tokens": out_tokens,
            "thinking_config": {"thinking_budget": 0},
        }
        if top_p is not None:
            config["top_p"] = top_p
        response = self._call(prompt, config)
        text = self._extract_text(response)
        text = self._strip_fences(text)
        text = self._extract_json(text)
        return text

    def sample_choice(
        self,
        prompt: str,
        responses: tuple[str, ...],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict]:
        forced = (
            prompt
            + "\n\nChoose exactly one option.\n"
            + "\n".join(f"{i}: {r}" for i, r in enumerate(responses))
            + "\n\nReply with only the index number."
        )
        config = {
            "temperature": 0.0,
            "max_output_tokens": 64,
            "thinking_config": {"thinking_budget": 0},
        }
        try:
            response = self._call(forced, config)
            text = (getattr(response, "text", None) or "").strip()
            m = re.search(r"\d+", text)
            if not m:
                return 0, text, {}
            idx = int(m.group(0))
            if not 0 <= idx < len(responses):
                idx = 0
            return idx, text, {}
        except Exception as e:
            print(f"[sample_choice] failed, defaulting to 0: {e}")
            return 0, "", {}


# ---------------------------------------------------------------------------
# Ray actor: one SentenceTransformer per node, shared across tasks
# ---------------------------------------------------------------------------

@ray.remote
class EmbedderActor:
    """Loads SentenceTransformer once per worker node and serves embed calls."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()


# ---------------------------------------------------------------------------
# Ray remote task: one trial
# ---------------------------------------------------------------------------

@ray.remote(max_retries=2, retry_exceptions=True)
def run_trial_ray(
    trial_id: int,
    seed: int,
    n_agents: int,
    max_steps: int,
    profiles_path_str: str,
    embedder_actor: ray.actor.ActorHandle,
) -> dict:
    """Execute one simulation trial inside a Ray worker."""
    # Re-add repo path in worker process
    repo_root = Path(profiles_path_str).resolve().parents[3]
    concordia_parent = repo_root / "concordia"
    if concordia_parent.exists() and str(concordia_parent) not in sys.path:
        sys.path.insert(0, str(concordia_parent))

    try:
        from cs_group_project_sim import (
            build_simulation,
            load_trait_pool,
            sample_profiles,
        )

        profiles = load_trait_pool(Path(profiles_path_str))
        sampled_profiles = sample_profiles(profiles, n_agents, seed)

        model = _GeminiModel(
            model_name="gemini-2.5-flash",
            api_key=os.environ["GEMINI_API_KEY"],
        )

        # Embedder via shared actor — track latency per call
        _embed_times: list[float] = []
        def embedder(text: str) -> list[float]:
            _t0 = time.perf_counter()
            result = ray.get(embedder_actor.embed.remote(text))
            _embed_times.append(time.perf_counter() - _t0)
            return result

        sim = build_simulation(
            sampled_profiles=sampled_profiles,
            model=model,
            embedder=embedder,
            max_steps=max_steps,
        )
        _trial_start = time.perf_counter()
        results = sim.play()
        _trial_elapsed = time.perf_counter() - _trial_start

        # --- Timing summary ---
        _gemini_times = model._call_times
        _n_gemini = len(_gemini_times)
        _n_embed = len(_embed_times)
        _gemini_total = sum(_gemini_times)
        _embed_total = sum(_embed_times)
        _gemini_avg = _gemini_total / _n_gemini if _n_gemini else 0
        _embed_avg = _embed_total / _n_embed if _n_embed else 0
        print(
            f"[Trial {trial_id}] total={_trial_elapsed:.1f}s | "
            f"gemini: {_n_gemini} calls, {_gemini_total:.1f}s total, {_gemini_avg:.2f}s avg | "
            f"embedder: {_n_embed} calls, {_embed_total:.1f}s total, {_embed_avg:.3f}s avg | "
            f"other: {max(0, _trial_elapsed - _gemini_total - _embed_total):.1f}s"
        )

        return {
            "trial_id": trial_id,
            "seed": seed,
            "status": "success",
            "profiles": [p["profile_id"] for p in sampled_profiles],
            "timing": {
                "trial_total_s": round(_trial_elapsed, 2),
                "gemini_calls": _n_gemini,
                "gemini_total_s": round(_gemini_total, 2),
                "gemini_avg_s": round(_gemini_avg, 3),
                "embed_calls": _n_embed,
                "embed_total_s": round(_embed_total, 2),
                "embed_avg_s": round(_embed_avg, 4),
                "other_s": round(max(0, _trial_elapsed - _gemini_total - _embed_total), 2),
            },
            "results": _serialize(results),
        }

    except Exception as e:
        return {
            "trial_id": trial_id,
            "seed": seed,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Serialization (same as original; duplicated here to avoid import issues
# in workers that may not have the full sim module on PATH yet)
# ---------------------------------------------------------------------------

def _clean_action(summary: str) -> str:
    """Strip Concordia scaffolding from the summary, leaving only the narrative."""
    import re
    text = summary.strip()
    # Remove "Step N course_staff --- " prefix
    text = re.sub("^Step [0-9]+ [^ ]+ --- ", "", text)
    # Remove any leading "Event: " repetitions
    text = re.sub("^(Event: *)+", "", text)
    return text.strip()


def _serialize(log, debug: bool = False) -> dict:
    """Extract one clean record per agent action: step, agent name, and narrative."""
    data: dict = {}

    # Per-step timeline: one entry per agent action (entry_type == "entity")
    try:
        steps_out: dict[str, list] = {}
        for step in log.get_steps():
            entries = log.get_entries_by_step(step)
            actions = []
            for e in entries:
                v = vars(e) if hasattr(e, "__dict__") else {}
                if v.get("entry_type") == "entity":
                    actions.append({
                        "agent": v.get("entity_name"),
                        "action": _clean_action(v.get("summary", "")),
                    })
            if actions:
                steps_out[str(step)] = actions
        data["steps"] = steps_out
    except Exception as e:
        data["steps_error"] = str(e)

    # Same actions grouped by agent
    try:
        names = log.get_entity_names()
        by_agent: dict[str, list] = {}
        for name in names:
            entries = log.get_entries_by_entity(name)
            actions = []
            for e in entries:
                v = vars(e) if hasattr(e, "__dict__") else {}
                if v.get("entry_type") == "entity":
                    actions.append({
                        "step": v.get("step"),
                        "action": _clean_action(v.get("summary", "")),
                    })
            if actions:
                by_agent[name] = actions
        data["actions_by_agent"] = by_agent
    except Exception as e:
        data["actions_by_agent_error"] = str(e)

    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallelised CS group project simulations via Ray."
    )
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/simulation_results.jsonl")
    parser.add_argument(
        "--max_concurrent", type=int, default=32,
        help="Max trials in-flight at once (tune to stay under Gemini RPM quota).",
    )
    parser.add_argument(
        "--address", type=str, default=None,
        help="Ray cluster address (e.g. 'auto' or 'ray://host:10001'). "
             "Omit to start a local cluster.",
    )
    parser.add_argument(
        "--num_cpus", type=int, default=None,
        help="CPUs for local Ray cluster. Defaults to all available.",
    )
    parser.add_argument(
        "--embedder_replicas", type=int, default=1,
        help="Number of EmbedderActor replicas (one per node is usually fine).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Init Ray
    # ------------------------------------------------------------------
    ray_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if args.address:
        ray_kwargs["address"] = args.address
    elif args.num_cpus:
        ray_kwargs["num_cpus"] = args.num_cpus
    ray.init(**ray_kwargs)
    print(f"Ray cluster resources: {ray.cluster_resources()}\n")

    # ------------------------------------------------------------------
    # Shared embedder actor(s)
    # ------------------------------------------------------------------
    embedder_actors = [
        EmbedderActor.options(name=f"embedder_{i}", get_if_exists=True).remote()
        for i in range(args.embedder_replicas)
    ]

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids: set[int] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed_ids.add(json.loads(line)["trial_id"])
                except Exception:
                    pass
        if completed_ids:
            print(f"Resuming — skipping {len(completed_ids)} already-completed trials.")

    seeds = [args.base_seed + i for i in range(args.n_trials)]
    pending = [i for i in range(args.n_trials) if i not in completed_ids]
    if not pending:
        print("All trials already complete.")
        ray.shutdown()
        return

    print(f"Launching {len(pending)} trials (max {args.max_concurrent} in-flight)...")
    print(f"Output: {output_path}\n")

    # ------------------------------------------------------------------
    # Sliding-window submission: keep max_concurrent futures in-flight
    # ------------------------------------------------------------------
    profiles_path_str = str(TRAIT_POOL_PATH)
    total = len(pending)
    ok = errors = 0
    pending_iter = iter(pending)

    # futures -> trial_id
    in_flight: dict[ray.ObjectRef, int] = {}

    def _submit_next() -> bool:
        try:
            i = next(pending_iter)
        except StopIteration:
            return False
        actor = embedder_actors[i % len(embedder_actors)]
        ref = run_trial_ray.remote(
            trial_id=i,
            seed=seeds[i],
            n_agents=args.n_agents,
            max_steps=args.max_steps,
            profiles_path_str=profiles_path_str,
            embedder_actor=actor,
        )
        in_flight[ref] = i
        return True

    # Fill initial window
    for _ in range(min(args.max_concurrent, total)):
        _submit_next()

    with open(output_path, "a", encoding="utf-8") as out_f:
        with tqdm(total=total, unit="trial") as pbar:
            while in_flight:
                # Wait for any one to finish
                done, _ = ray.wait(list(in_flight.keys()), num_returns=1, timeout=5.0)
                if not done:
                    continue

                ref = done[0]
                in_flight.pop(ref)
                result: dict = ray.get(ref)

                # Write immediately
                out_f.write(json.dumps(result, default=str) + "\n")
                out_f.flush()

                if result["status"] == "error":
                    errors += 1
                    tqdm.write(f"[Trial {result['trial_id']}] ERROR: {result['error']}")
                else:
                    ok += 1

                pbar.update(1)
                pbar.set_postfix({"ok": ok, "err": errors, "in_flight": len(in_flight)})

                # Refill window
                _submit_next()

    print(f"\nDone. {ok} succeeded, {errors} failed.")
    print(f"Results written to: {output_path}")
    ray.shutdown()


if __name__ == "__main__":
    main()