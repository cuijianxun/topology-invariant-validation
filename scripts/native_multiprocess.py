"""Native OS-process topology experiment for topology (not PyTorch/Gloo)."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gate", ROOT / "scripts" / "topology_checker.py")
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def worker(payload: tuple[list[str], int]) -> dict:
    tokens, master_seed = payload
    return {
        "pid": os.getpid(),
        "rows": [GATE.row(token, master_seed) for token in tokens],
    }


def run_native(repeats: int = 10) -> dict:
    context = mp.get_context("spawn")
    cases = []
    distinct_child_pids: set[int] = set()
    for world_size in GATE.WORLD_SIZES:
        with context.Pool(processes=world_size) as pool:
            for token_count in GATE.TOKEN_COUNTS:
                tokens = [f"native_{index:03d}_{'x' * (index % 5)}" for index in range(token_count)]
                baseline = GATE.signature(GATE.correct_evaluate(tokens, 1, "contiguous", GATE.MASTER_SEED))
                for mode in GATE.SHARDINGS:
                    durations = []
                    relation_passes = []
                    observed_pids: set[int] = set()
                    for _ in range(repeats):
                        parts = GATE.shard(tokens, world_size, mode)
                        started = time.perf_counter()
                        results = pool.map(worker, [(part, GATE.MASTER_SEED) for part in parts])
                        durations.append(time.perf_counter() - started)
                        rows = [item for result in results for item in result["rows"]]
                        relation_passes.append(GATE.signature(rows) == baseline)
                        observed_pids.update(result["pid"] for result in results)
                    distinct_child_pids.update(observed_pids)
                    cases.append(
                        {
                            "token_count": token_count,
                            "world_size": world_size,
                            "sharding": mode,
                            "repeats": repeats,
                            "all_relations_pass": all(relation_passes),
                            "observed_child_pids": sorted(observed_pids),
                            "median_seconds": statistics.median(durations),
                            "p95_seconds": sorted(durations)[max(0, int(0.95 * len(durations)) - 1)],
                        }
                    )
    gates = {
        "all_24_topologies_pass": len(cases) == 24 and all(case["all_relations_pass"] for case in cases),
        "true_child_processes_observed": len(distinct_child_pids) >= 2,
        "ten_repeats_each": repeats >= 10 and all(case["repeats"] >= 10 for case in cases),
    }
    return {
        "evidence_level": "NATIVE_MULTIPROCESS_NOT_GLOO",
        "cases": cases,
        "distinct_child_pids": sorted(distinct_child_pids),
        "gates": gates,
        "overall_pass": all(gates.values()),
        "gloo_gate_status": "BLOCKED_MISSING_TORCH",
    }


def main() -> None:
    summary = run_native()
    output = ROOT / "results" / "evaluation" / "native_multiprocess_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in summary.items() if key != "cases"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
