"""Run N simulation trials in parallel and write all outputs to a JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from cs_group_project_sim import (
    load_trait_pool,
    sample_profiles,
    build_simulation,
    TRAIT_POOL_PATH,
)


def serialize_log(log) -> dict:
    """Extract all useful data from a Concordia SimulationLog object."""
    data = {}

    # Full structured data
    try:
        data["log"] = log.to_dict()
    except Exception as e:
        data["log_error"] = str(e)

    # Human-readable summary
    try:
        data["summary"] = log.get_summary()
    except Exception as e:
        data["summary_error"] = str(e)

    # Per-entity memories
    try:
        entity_names = log.get_entity_names()
        data["entity_memories"] = {
            name: log.get_entity_memories(name) for name in entity_names
        }
    except Exception as e:
        data["entity_memories_error"] = str(e)

    # Game master memories
    try:
        data["game_master_memories"] = log.get_game_master_memories()
    except Exception as e:
        data["game_master_memories_error"] = str(e)

    # All entries grouped by step
    try:
        data["steps"] = {}
        for step in log.get_steps():
            entries = log.get_entries_by_step(step)
            data["steps"][str(step)] = [
                json.loads(json.dumps(e, default=str)) for e in entries
            ]
    except Exception as e:
        data["steps_error"] = str(e)

    # Entries grouped by entity
    try:
        entity_names = log.get_entity_names()
        data["entries_by_entity"] = {
            name: [
                json.loads(json.dumps(e, default=str))
                for e in log.get_entries_by_entity(name)
            ]
            for name in entity_names
        }
    except Exception as e:
        data["entries_by_entity_error"] = str(e)

    return data


def run_single_trial(
    trial_id: int,
    seed: int,
    n_agents: int,
    max_steps: int,
    profiles_path: Path,
) -> dict:
    """Run one simulation trial. Executed in a worker process."""
    try:
        from cs_group_project_sim import get_model_and_embedder

        profiles = load_trait_pool(profiles_path)
        sampled_profiles = sample_profiles(profiles, n_agents, seed)
        model, embedder = get_model_and_embedder()

        sim = build_simulation(
            sampled_profiles=sampled_profiles,
            model=model,
            embedder=embedder,
            max_steps=max_steps,
        )
        results = sim.play()

        return {
            "trial_id": trial_id,
            "seed": seed,
            "status": "success",
            "profiles": [p["profile_id"] for p in sampled_profiles],
            "results": serialize_log(results),
        }

    except Exception as e:
        return {
            "trial_id": trial_id,
            "seed": seed,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def write_result(result: dict, output_path: Path, lock: Lock) -> None:
    """Append one result as a JSON line. Lock ensures no interleaving."""
    with lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/simulation_results.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip trials already in the output file
    completed_ids: set[int] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    completed_ids.add(r["trial_id"])
                except Exception:
                    pass
        if completed_ids:
            print(f"Resuming — skipping {len(completed_ids)} already-completed trials.")

    seeds = [args.base_seed + i for i in range(args.n_trials)]
    pending = [i for i in range(args.n_trials) if i not in completed_ids]

    if not pending:
        print("All trials already complete.")
        return

    lock = Lock()
    completed = 0
    errors = 0

    print(f"Launching {len(pending)} trials across {args.max_workers} workers...")
    print(f"Output: {output_path}\n")

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_single_trial,
                trial_id=i,
                seed=seeds[i],
                n_agents=args.n_agents,
                max_steps=args.max_steps,
                profiles_path=TRAIT_POOL_PATH,
            ): i
            for i in pending
        }

        with tqdm(total=len(pending), unit="trial") as pbar:
            for future in as_completed(futures):
                result = future.result()
                write_result(result, output_path, lock)

                if result["status"] == "error":
                    errors += 1
                    tqdm.write(f"[Trial {result['trial_id']}] ERROR: {result['error']}")
                else:
                    completed += 1

                pbar.update(1)
                pbar.set_postfix({"ok": completed, "err": errors})

    print(f"\nDone. {completed} succeeded, {errors} failed.")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()