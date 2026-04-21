"""
run_parallel_ray.py
-------------------
Run all 100 simulation trials in parallel using Ray, for any condition.

Each trial N uses group N from configs/simulation_groups.jsonl so every
condition runs the exact same 100 pre-sampled teams. This makes results
directly comparable across conditions without any sampling variability.

Parallelism strategy
--------------------
Each trial runs as a Ray remote task. A shared EmbedderActor per node
loads SentenceTransformer once instead of once per trial.
Gemini calls use the sync google.genai SDK — no aiohttp, no event-loop
conflicts with Concordia's internal asyncio concurrency.

Usage
-----
    # Single condition (100 trials):
    python run_parallel_ray.py --condition control
    python run_parallel_ray.py --condition weekly_log

    # All 7 conditions sequentially (700 trials total):
    python run_parallel_ray.py --condition all

    # Multi-node (start Ray head first):
    ray start --head --port 6379
    python run_parallel_ray.py --address auto --condition all

    # Tune concurrency to stay under Gemini RPM quota:
    python run_parallel_ray.py --max_concurrent 16

Available conditions:
    control  contribution_tracking  task_visibility  peer_evaluation
    weekly_log  meaningful_feedback  agile  all
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
# Repo / package path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_CONCORDIA_PARENT = _REPO_ROOT / "concordia"
if _LOCAL_CONCORDIA_PARENT.exists():
    sys.path.insert(0, str(_LOCAL_CONCORDIA_PARENT))

from cs_group_project_sim import SIMULATION_GROUPS_PATH
from interventions import REGISTRY, ALL_CONDITION_NAMES


# ---------------------------------------------------------------------------
# Gemini model (self-contained for Ray workers)
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
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self._call_times: list[float] = []

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
        # Safety filter or empty output — return neutral fallback so
        # Concordia can continue rather than crash.
        for cand in candidates:
            finish = str(getattr(cand, "finish_reason", "")).upper()
            if finish in ("STOP", "SAFETY", "RECITATION", "OTHER", ""):
                return "None."
        raise RuntimeError(f"Gemini returned no usable text: {response!r}")

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()

    @staticmethod
    def _sanitize_action_spec(text: str) -> str:
        """Repair unescaped double quotes inside call_to_action JSON values."""
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
        plain = raw_value.replace('\\"', '"')
        plain = plain.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        clean = plain.replace('\\', '\\\\').replace('"', '\\"')
        return text[:value_start] + clean + text[end_idx:]

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return text
        candidate = m.group(0).strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
        try:
            repaired = _GeminiModel._sanitize_action_spec(candidate)
            json.loads(repaired)
            return repaired
        except Exception:
            pass
        return text

    def _call(self, contents: str, config: dict) -> Any:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                result = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                self._call_times.append(time.perf_counter() - t0)
                return result
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 60)
                    print(f"[Gemini] attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        raise RuntimeError(
            f"Gemini call failed after {self.max_retries} retries: {last_err}"
        )

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
            self.max_output_tokens if max_tokens is None else max_tokens, 2048
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
        return self._extract_json(text)

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
# Shared embedder actor — one SentenceTransformer per node
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
# Log serialization
# ---------------------------------------------------------------------------

def _clean_action(summary: str) -> str:
    text = summary.strip()
    text = re.sub(r"^Step [0-9]+ [^ ]+ --- ", "", text)
    text = re.sub(r"^(Event: *)+", "", text)
    return text.strip()


def _serialize(log: Any) -> dict:
    """
    Extract two views from a Concordia SimulationLog:
      - steps:            {step_number: [{agent, action}, ...]}
      - actions_by_agent: {agent_name:  [{step, action}, ...]}
    Only entity-type entries (actual agent actions) are included.
    Each extraction is attempted independently so a single failure does
    not discard the rest of the data.
    """
    data: dict = {}

    try:
        steps_out: dict[str, list] = {}
        for step in log.get_steps():
            actions = []
            for e in log.get_entries_by_step(step):
                v = vars(e) if hasattr(e, "__dict__") else {}
                if v.get("entry_type") == "entity":
                    actions.append({
                        "agent":  v.get("entity_name"),
                        "action": _clean_action(v.get("summary", "")),
                    })
            if actions:
                steps_out[str(step)] = actions
        data["steps"] = steps_out
    except Exception as e:
        data["steps_error"] = str(e)

    try:
        by_agent: dict[str, list] = {}
        for name in log.get_entity_names():
            actions = []
            for e in log.get_entries_by_entity(name):
                v = vars(e) if hasattr(e, "__dict__") else {}
                if v.get("entry_type") == "entity":
                    actions.append({
                        "step":   v.get("step"),
                        "action": _clean_action(v.get("summary", "")),
                    })
            if actions:
                by_agent[name] = actions
        data["actions_by_agent"] = by_agent
    except Exception as e:
        data["actions_by_agent_error"] = str(e)

    return data


# ---------------------------------------------------------------------------
# Ray remote task — one trial
# ---------------------------------------------------------------------------

@ray.remote(max_retries=2, retry_exceptions=True)
def run_trial_ray(
    group_index: int,
    max_steps: int,
    groups_path_str: str,
    embedder_actor: ray.actor.ActorHandle,
    condition_name: str = "control",
) -> dict:
    """Execute one simulation trial inside a Ray worker process.

    group_index identifies which of the 100 pre-sampled groups to use.
    The same group_index is used for every condition, ensuring all
    conditions are compared on identical agent compositions.
    """
    repo_root = Path(groups_path_str).resolve().parents[3]
    concordia_parent = repo_root / "concordia"
    if concordia_parent.exists() and str(concordia_parent) not in sys.path:
        sys.path.insert(0, str(concordia_parent))

    try:
        from cs_group_project_sim import (
            build_simulation,
            load_simulation_groups,
            get_group_agents,
        )
        from interventions import REGISTRY

        condition   = REGISTRY[condition_name]
        groups      = load_simulation_groups(Path(groups_path_str))
        agents      = get_group_agents(groups, group_index)
        group_id    = groups[group_index]["group_id"]

        model = _GeminiModel(
            model_name="gemini-2.5-flash",
            api_key=os.environ["GEMINI_API_KEY"],
        )

        embed_times: list[float] = []
        def embedder(text: str) -> list[float]:
            t0 = time.perf_counter()
            result = ray.get(embedder_actor.embed.remote(text))
            embed_times.append(time.perf_counter() - t0)
            return result

        sim = build_simulation(
            agents=agents,
            model=model,
            embedder=embedder,
            max_steps=max_steps,
            condition=condition,
            group_id=group_id,
        )
        t_start = time.perf_counter()
        results = sim.play()
        trial_elapsed = time.perf_counter() - t_start

        gemini_times = model._call_times
        gemini_total = sum(gemini_times)
        embed_total  = sum(embed_times)
        n_gemini     = len(gemini_times)
        n_embed      = len(embed_times)

        print(
            f"[{group_id} | {condition_name}] "
            f"total={trial_elapsed:.1f}s | "
            f"gemini: {n_gemini} calls, {gemini_total:.1f}s total, "
            f"{gemini_total/n_gemini if n_gemini else 0:.2f}s avg | "
            f"embedder: {n_embed} calls, {embed_total:.1f}s total"
        )

        return {
            "group_index": group_index,
            "group_id":    group_id,
            "condition":   condition_name,
            "status":      "success",
            "profiles":    [a["profile_id"] for a in agents],
            "timing": {
                "trial_total_s":  round(trial_elapsed, 2),
                "gemini_calls":   n_gemini,
                "gemini_total_s": round(gemini_total, 2),
                "gemini_avg_s":   round(gemini_total / n_gemini if n_gemini else 0, 3),
                "embed_calls":    n_embed,
                "embed_total_s":  round(embed_total, 2),
                "embed_avg_s":    round(embed_total / n_embed if n_embed else 0, 4),
                "other_s":        round(
                    max(0, trial_elapsed - gemini_total - embed_total), 2
                ),
            },
            "results": _serialize(results),
        }

    except Exception as e:
        return {
            "group_index": group_index,
            "condition":   condition_name,
            "status":      "error",
            "error":       str(e),
            "traceback":   traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_completed_keys(output_path: Path) -> set[tuple[int, str]]:
    """Return (group_index, condition) pairs already written to the output file."""
    completed: set[tuple[int, str]] = set()
    if not output_path.exists():
        return completed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add((rec["group_index"], rec.get("condition", "control")))
            except Exception:
                pass
    return completed


# ---------------------------------------------------------------------------
# Core runner — one condition
# ---------------------------------------------------------------------------

def run_condition(
    condition_name: str,
    n_groups: int,
    max_steps: int,
    max_concurrent: int,
    output_path: Path,
    embedder_actors: list,
    groups_path_str: str,
) -> tuple[int, int]:
    """Run all trials for one condition. Returns (n_ok, n_errors)."""
    completed_keys = load_completed_keys(output_path)
    pending = [
        i for i in range(n_groups)
        if (i, condition_name) not in completed_keys
    ]

    if not pending:
        print(f"[{condition_name}] All {n_groups} trials already complete, skipping.")
        return 0, 0

    skipped = n_groups - len(pending)
    if skipped:
        print(f"[{condition_name}] Resuming — skipping {skipped} completed trials.")

    ok = errors = 0
    pending_iter = iter(pending)
    in_flight: dict[ray.ObjectRef, int] = {}

    def _submit_next() -> bool:
        try:
            i = next(pending_iter)
        except StopIteration:
            return False
        actor = embedder_actors[i % len(embedder_actors)]
        ref = run_trial_ray.remote(
            group_index=i,
            max_steps=max_steps,
            groups_path_str=groups_path_str,
            embedder_actor=actor,
            condition_name=condition_name,
        )
        in_flight[ref] = i
        return True

    for _ in range(min(max_concurrent, len(pending))):
        _submit_next()

    with open(output_path, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(pending), unit="trial", desc=condition_name) as pbar:
            while in_flight:
                done, _ = ray.wait(list(in_flight.keys()), num_returns=1, timeout=5.0)
                if not done:
                    continue
                ref = done[0]
                in_flight.pop(ref)
                result: dict = ray.get(ref)

                out_f.write(json.dumps(result, default=str) + "\n")
                out_f.flush()

                if result["status"] == "error":
                    errors += 1
                    tqdm.write(
                        f"[{result.get('group_id', result['group_index'])} | "
                        f"{condition_name}] ERROR: {result['error']}"
                    )
                else:
                    ok += 1

                pbar.update(1)
                pbar.set_postfix({"ok": ok, "err": errors, "in_flight": len(in_flight)})
                _submit_next()

    return ok, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallelised CS group-project simulations via Ray.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="control",
        choices=ALL_CONDITION_NAMES + ["all"],
        help=(
            "Condition to run. Pass 'all' to run every condition sequentially "
            f"({', '.join(ALL_CONDITION_NAMES)})."
        ),
    )
    parser.add_argument("--max_steps",      type=int, default=20)
    parser.add_argument("--max_concurrent", type=int, default=32,
        help="Max trials in-flight at once (tune to stay under Gemini RPM quota).")
    parser.add_argument("--output", type=str,
        default="results/simulation_results.json")
    parser.add_argument(
        "--groups_path", type=str, default="",
        help="Override path to simulation_groups.json.",
    )
    parser.add_argument(
        "--address", type=str, default=None,
        help="Ray cluster address (e.g. 'auto'). Omit to start a local cluster.",
    )
    parser.add_argument("--num_cpus", type=int, default=None,
        help="CPUs for local Ray cluster. Defaults to all available.")
    parser.add_argument("--embedder_replicas", type=int, default=1,
        help="Number of EmbedderActor replicas (one per node is usually fine).")
    args = parser.parse_args()

    conditions_to_run = (
        ALL_CONDITION_NAMES if args.condition == "all" else [args.condition]
    )
    groups_path_str = args.groups_path or str(SIMULATION_GROUPS_PATH)
    groups_path = Path(groups_path_str)

    # Count groups in file to determine n_groups
    with open(groups_path) as f:
        n_groups = sum(1 for line in f if line.strip())
    print(f"Loaded {n_groups} groups from {groups_path}")

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
    # Output file
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run each condition
    # ------------------------------------------------------------------
    total_ok = total_errors = 0

    for condition_name in conditions_to_run:
        print(f"\n{'='*60}")
        print(f"Condition : {REGISTRY[condition_name].label}")
        print(f"Groups    : {n_groups}  |  Max concurrent: {args.max_concurrent}")
        print(f"Output    : {output_path}")
        print('='*60)

        ok, errors = run_condition(
            condition_name=condition_name,
            n_groups=n_groups,
            max_steps=args.max_steps,
            max_concurrent=args.max_concurrent,
            output_path=output_path,
            embedder_actors=embedder_actors,
            groups_path_str=groups_path_str,
        )
        total_ok     += ok
        total_errors += errors
        print(f"[{condition_name}] {ok} succeeded, {errors} failed.")

    print(f"\n{'='*60}")
    print(f"All done. Total: {total_ok} succeeded, {total_errors} failed.")
    print(f"Results written to: {output_path}")
    ray.shutdown()


if __name__ == "__main__":
    main()