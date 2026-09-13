"""CPU scalability benchmark for the topology trace checker."""

from __future__ import annotations

import gc
import importlib.util
import json
import statistics
import time
import tracemalloc
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gate", ROOT / "scripts" / "topology_checker.py")
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * len(ordered)) - 1))
    return ordered[index]


def run_scalability(repeats: int = 30) -> dict:
    cases = []
    for token_count in (100, 1_000, 10_000, 100_000):
        tokens = [f"scale_{index:06d}" for index in range(token_count)]
        canonical_rows = [GATE.row(token, GATE.MASTER_SEED) for token in tokens]
        row_by_token = {row["token"]: row for row in canonical_rows}
        baseline = GATE.signature(canonical_rows)
        for world_size in (1, 2, 4, 8):
            parts = GATE.shard(tokens, world_size, "round_robin")
            ordered_tokens = [token for part in parts for token in part]
            durations = []
            checks = []
            for _ in range(repeats):
                started = time.perf_counter()
                observed = GATE.signature([row_by_token[token] for token in ordered_tokens])
                durations.append(time.perf_counter() - started)
                checks.append(observed == baseline)

            gc.collect()
            tracemalloc.start()
            observed = GATE.signature([row_by_token[token] for token in ordered_tokens])
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            cases.append(
                {
                    "token_count": token_count,
                    "world_size": world_size,
                    "repeats": repeats,
                    "all_relations_pass": all(checks) and observed == baseline,
                    "median_seconds": statistics.median(durations),
                    "iqr_seconds": percentile(durations, 0.75) - percentile(durations, 0.25),
                    "p95_seconds": percentile(durations, 0.95),
                    "peak_tracemalloc_bytes": peak_bytes,
                    "microseconds_per_token_median": statistics.median(durations) * 1_000_000 / token_count,
                }
            )

    gates = {
        "four_token_scales": {case["token_count"] for case in cases} == {100, 1_000, 10_000, 100_000},
        "world_sizes_1_2_4_8": {case["world_size"] for case in cases} == {1, 2, 4, 8},
        "thirty_repeats": repeats >= 30 and all(case["repeats"] >= 30 for case in cases),
        "all_relations_pass": all(case["all_relations_pass"] for case in cases),
    }
    return {
        "schema_version": 1,
        "evidence_level": "CPU_CHECKER_SCALABILITY_NOT_MODEL_RUNTIME",
        "cases": cases,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def main() -> None:
    result = run_scalability()
    output = ROOT / "results" / "evaluation" / "scalability_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
