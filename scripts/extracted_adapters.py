"""Native-process reproductions of three commit-pinned evaluation semantics.

These adapters preserve orchestration semantics while replacing models and datasets
with deterministic token fixtures. They are not executions of the original pipelines.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gate", ROOT / "scripts" / "topology_checker.py")
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def _token_worker(payload: tuple[int, list[str], int]) -> dict:
    rank, tokens, master_seed = payload
    return {
        "pid": os.getpid(),
        "rank": rank,
        "rows": [GATE.row(token, master_seed) for token in tokens],
    }


def _payload_worker(payload: tuple[int, list[str], int]) -> dict:
    result = _token_worker(payload)
    envelope = {
        "rank": result["rank"],
        "rows": result["rows"],
        # Make every nonzero-rank payload larger even when rank 0 owns one
        # extra token. This deterministically exercises the pinned trim bug.
        "rank_dependent_padding": "x" * (4096 * result["rank"]),
    }
    result["blob"] = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def _signature(rows: list[dict]) -> dict:
    return GATE.signature(rows)


def _relation(rows: list[dict], baseline: dict) -> bool:
    return _signature(rows) == baseline


def drivedreamer_adapter(
    pool: mp.pool.Pool, tokens: list[str], world_size: int, repaired: bool
) -> tuple[list[dict], set[int], str | None]:
    parts = GATE.shard(tokens, world_size, "contiguous")
    results = pool.map(_token_worker, [(rank, part, GATE.MASTER_SEED) for rank, part in enumerate(parts)])
    rows = [row for result in (results if repaired else results[:1]) for row in result["rows"]]
    return rows, {result["pid"] for result in results}, None


def drivelaw_adapter(
    pool: mp.pool.Pool, tokens: list[str], world_size: int, repaired: bool
) -> tuple[list[dict], set[int], str | None]:
    parts = GATE.shard(tokens, world_size, "round_robin")
    results = pool.map(_payload_worker, [(rank, part, GATE.MASTER_SEED) for rank, part in enumerate(parts)])
    sizes = [len(result["blob"]) for result in results]
    max_size = max(sizes)
    padded = [result["blob"] + bytes(max_size - len(result["blob"])) for result in results]
    decoded = []
    try:
        for rank, blob in enumerate(padded):
            trim_size = sizes[rank] if repaired else sizes[0]
            decoded.append(pickle.loads(blob[:trim_size]))
    except (EOFError, pickle.UnpicklingError) as exc:
        return [], {result["pid"] for result in results}, type(exc).__name__
    rows = [row for envelope in decoded for row in envelope["rows"]]
    return rows, {result["pid"] for result in results}, None


def opendwm_adapter(
    pool: mp.pool.Pool,
    tokens: list[str],
    world_size: int,
    repaired: bool,
    batch_size: int = 4,
) -> tuple[list[dict], set[int], str | None]:
    if repaired:
        selected = tokens
    else:
        length = len(tokens) // (world_size * batch_size) * (world_size * batch_size)
        selected = tokens[:length]
    parts = GATE.shard(selected, world_size, "contiguous")
    results = pool.map(_token_worker, [(rank, part, GATE.MASTER_SEED) for rank, part in enumerate(parts)])
    rows = [row for result in results for row in result["rows"]]
    return rows, {result["pid"] for result in results}, None


ADAPTERS = {
    "DriveDreamer_prepared_shard_main_only": drivedreamer_adapter,
    "DriveLaW_rank0_size_for_all_payloads": drivelaw_adapter,
    "OpenDWM_world_batch_floor": opendwm_adapter,
}


def run_adapters(repeats: int = 10) -> dict:
    context = mp.get_context("spawn")
    cases = []
    all_pids: set[int] = set()
    for world_size in (2, 3, 4):
        with context.Pool(processes=world_size) as pool:
            for token_count in (17, 23):
                tokens = [f"adapter_{index:03d}_{'z' * (index % 7)}" for index in range(token_count)]
                baseline = _signature(GATE.correct_evaluate(tokens, 1, "contiguous", GATE.MASTER_SEED))
                for adapter_name, adapter in ADAPTERS.items():
                    faithful_violations = []
                    repaired_passes = []
                    errors = []
                    case_pids: set[int] = set()
                    for _ in range(repeats):
                        faithful_rows, pids, faithful_error = adapter(pool, tokens, world_size, False)
                        repaired_rows, repaired_pids, repaired_error = adapter(pool, tokens, world_size, True)
                        case_pids.update(pids | repaired_pids)
                        faithful_violations.append(
                            faithful_error is not None or not _relation(faithful_rows, baseline)
                        )
                        repaired_passes.append(
                            repaired_error is None and _relation(repaired_rows, baseline)
                        )
                        errors.append({"faithful": faithful_error, "repaired": repaired_error})
                    all_pids.update(case_pids)
                    cases.append(
                        {
                            "adapter": adapter_name,
                            "world_size": world_size,
                            "token_count": token_count,
                            "repeats": repeats,
                            "faithful_violation_runs": sum(faithful_violations),
                            "repaired_pass_runs": sum(repaired_passes),
                            "observed_child_pids": sorted(case_pids),
                            "errors": errors,
                        }
                    )

    gates = {
        "three_adapters": len(ADAPTERS) == 3,
        "three_world_sizes": {case["world_size"] for case in cases} == {2, 3, 4},
        "faithful_triggers_every_run": all(
            case["faithful_violation_runs"] == case["repeats"] for case in cases
        ),
        "repaired_passes_every_run": all(
            case["repaired_pass_runs"] == case["repeats"] for case in cases
        ),
        "native_child_processes": len(all_pids) >= 2,
    }
    return {
        "schema_version": 1,
        "evidence_level": "EXTRACTED_SEMANTICS_NATIVE_MULTIPROCESS_NOT_ORIGINAL_PIPELINE_NOT_GLOO",
        "source_commits": {
            "DriveDreamer": "da1ca92f831bc23d91b59ad418eb47b41cbb1fa9",
            "DriveLaW": "b26bffe57dae9e874512818194b204006ab0d089",
            "OpenDWM": "b0ecc3d4020612376ea5a87500f98bc76893428f",
        },
        "cases": cases,
        "distinct_child_pids": sorted(all_pids),
        "gates": gates,
        "overall_pass": all(gates.values()),
        "gloo_gate_status": "SEPARATE_NATIVE_GLOO_PASS",
    }


def main() -> None:
    summary = run_adapters()
    output = ROOT / "results" / "evaluation" / "extracted_adapters_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
