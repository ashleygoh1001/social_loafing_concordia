"""
run_parallel.py
---------------
Run all 100 simulation trials in parallel using ProcessPoolExecutor.

Each trial N uses group N from configs/simulation_groups.jsonl so every
condition runs the exact same 100 pre-sampled teams. This makes results
directly comparable across conditions without any sampling variability.

Use run_parallel_ray.py for large runs (shared embedder actor, better
resume support, multi-node). This file requires no Ray installation.

Usage
-----
    # Single condition:
    python run_parallel.py --condition control
    python run_parallel.py --condition weekly_log

    # All 7 conditions sequentially (700 trials total):
    python run_parallel.py --condition all

    # Tune workers to stay under Gemini RPM quota:
    python run_parallel.py --max_workers 8

Available conditions:
    control  contribution_tracking  task_visibility  peer_evaluation
    weekly_log  meaningful_feedback  agile  all
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_CONCORDIA_PARENT = _REPO_ROOT / "concordia"
if _LOCAL_CONCORDIA_PARENT.exists():
    sys.path.insert(0, str(_LOCAL_CONCORDIA_PARENT))

from cs_group_project_sim import SIMULATION_GROUPS_PATH
from interventions import REGISTRY, ALL_CONDITION_NAMES


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_single_trial(
    group_index: int,
    max_steps: int,
    groups_path: Path,
    condition_name: str,
) -> dict:
    """Run one simulation trial in a worker process.

    group_index maps directly to one of the 100 pre-sampled groups.
    """
    try:
        from cs_group_project_sim import (
            build_simulation,
            get_model_and_embedder,
            load_simulation_groups,
            get_group_agents,
        )
        from interventions import REGISTRY

        condition = REGISTRY[condition_name]
        groups    = load_simulation_groups(groups_path)
        agents    = get_group_agents(groups, group_index)
        group_id  = groups[group_index]["group_id"]

        model, embedder = get_model_and_embedder()

        sim = build_simulation(
            agents=agents,
            model=model,
            embedder=embedder,
            max_steps=max_steps,
            condition=condition,
            group_id=group_id,
        )
        results = sim.play()

        return {
            "group_index": group_index,
            "group_id":    group_id,
            "condition":   condition_name,
            "status":      "success",
            "profiles":    [a["profile_id"] for a in agents],
            "results":     _serialize(results),
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
# Log serialization
# ---------------------------------------------------------------------------

def _serialize(log) -> dict:
    import re

    def _clean(summary: str) -> str:
        text = summary.strip()
        text = re.sub(r"^Step [0-9]+ [^ ]+ --- ", "", text)
        text = re.sub(r"^(Event: *)+", "", text)
        return text.strip()

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
                        "action": _clean(v.get("summary", "")),
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
                        "action": _clean(v.get("summary", "")),
                    })
            if actions:
                by_agent[name] = actions
        data["actions_by_agent"] = by_agent
    except Exception as e:
        data["actions_by_agent_error"] = str(e)

    return data


# ---------------------------------------------------------------------------
# Resume / write helpers
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


def write_result(result: dict, output_path: Path, lock: Lock) -> None:
    with lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=str) + "\n")


# ---------------------------------------------------------------------------
# Core runner — one condition
# ---------------------------------------------------------------------------

def run_condition(
    condition_name: str,
    n_groups: int,
    max_steps: int,
    max_workers: int,
    output_path: Path,
    groups_path: Path,
) -> tuple[int, int]:
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

    lock = Lock()
    ok = errors = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_trial,
                group_index=i,
                max_steps=max_steps,
                groups_path=groups_path,
                condition_name=condition_name,
            ): i
            for i in pending
        }

        with tqdm(total=len(pending), unit="trial", desc=condition_name) as pbar:
            for future in as_completed(futures):
                result = future.result()
                write_result(result, output_path, lock)

                if result["status"] == "error":
                    errors += 1
                    tqdm.write(
                        f"[group_{result['group_index']:03d} | {condition_name}] "
                        f"ERROR: {result['error']}"
                    )
                else:
                    ok += 1

                pbar.update(1)
                pbar.set_postfix({"ok": ok, "err": errors})

    return ok, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CS group-project simulations in parallel (ProcessPoolExecutor).",
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
    parser.add_argument("--max_steps",   type=int, default=20)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--output", type=str,
        default="results/simulation_results.json")
    parser.add_argument(
        "--groups_path", type=str, default="",
        help="Override path to simulation_groups.json.",
    )
    args = parser.parse_args()

    conditions_to_run = (
        ALL_CONDITION_NAMES if args.condition == "all" else [args.condition]
    )
    groups_path = Path(args.groups_path) if args.groups_path else SIMULATION_GROUPS_PATH

    # Count groups in file
    with open(groups_path) as f:
        n_groups = sum(1 for line in f if line.strip())
    print(f"Loaded {n_groups} groups from {groups_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_ok = total_errors = 0

    for condition_name in conditions_to_run:
        print(f"\n{'='*60}")
        print(f"Condition : {REGISTRY[condition_name].label}")
        print(f"Groups    : {n_groups}  |  Workers: {args.max_workers}")
        print(f"Output    : {output_path}")
        print('='*60)

        ok, errors = run_condition(
            condition_name=condition_name,
            n_groups=n_groups,
            max_steps=args.max_steps,
            max_workers=args.max_workers,
            output_path=output_path,
            groups_path=groups_path,
        )
        total_ok     += ok
        total_errors += errors
        print(f"[{condition_name}] {ok} succeeded, {errors} failed.")

    print(f"\n{'='*60}")
    print(f"All done. Total: {total_ok} succeeded, {total_errors} failed.")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()