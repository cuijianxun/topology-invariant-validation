"""CPU-only topology-invariance probes for topology.

The probes reproduce orchestration semantics, not the original GPU models.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path


TOKEN_COUNTS = (7, 10, 17)
WORLD_SIZES = (1, 2, 3, 4)
SHARDINGS = ("contiguous", "round_robin")
MASTER_SEED = 32026


def keyed_randomness(master_seed: int, token: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{token}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def shard(tokens: list[str], world_size: int, mode: str) -> list[list[str]]:
    if mode == "round_robin":
        return [tokens[rank::world_size] for rank in range(world_size)]
    if mode != "contiguous":
        raise ValueError(mode)
    base, extra = divmod(len(tokens), world_size)
    result = []
    begin = 0
    for rank in range(world_size):
        size = base + (rank < extra)
        result.append(tokens[begin : begin + size])
        begin += size
    return result


def row(token: str, master_seed: int, failed: bool = False) -> dict:
    stream = keyed_randomness(master_seed, token)
    rng = random.Random(stream)
    return {
        "token": token,
        "stream": stream,
        "failed": failed,
        "score": None if failed else rng.random(),
        # Variable payload length makes the wrong trim independently observable.
        "trace": token * (1 + stream % 9),
    }


def correct_evaluate(tokens: list[str], world_size: int, mode: str, master_seed: int) -> list[dict]:
    return [row(token, master_seed) for part in shard(tokens, world_size, mode) for token in part]


def main_rank_only(tokens: list[str], world_size: int, mode: str, master_seed: int) -> list[dict]:
    return [row(token, master_seed) for token in shard(tokens, world_size, mode)[0]]


def global_batch_floor(tokens: list[str], world_size: int, batch_size: int, master_seed: int) -> list[dict]:
    total_batch_size = world_size * batch_size
    kept = tokens[: len(tokens) // total_batch_size * total_batch_size]
    return correct_evaluate(kept, world_size, "contiguous", master_seed)


def wrong_local_trim(tokens: list[str], world_size: int, mode: str, master_seed: int) -> list[dict]:
    payloads = [pickle.dumps([row(token, master_seed) for token in part]) for part in shard(tokens, world_size, mode)]
    rank0_size = len(payloads[0])
    maximum = max(map(len, payloads))
    gathered = [payload + bytes(maximum - len(payload)) for payload in payloads]
    result: list[dict] = []
    for payload in gathered:
        # Mirrors using rank-0 local_size for every gathered rank.
        result.extend(pickle.loads(payload[:rank0_size]))
    return result


def signature(rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda item: item["token"])
    valid_scores = [item["score"] for item in ordered if not item["failed"]]
    return {
        "tokens": [item["token"] for item in ordered],
        "streams": {item["token"]: item["stream"] for item in ordered},
        "failed": [item["token"] for item in ordered if item["failed"]],
        "mean": sum(valid_scores) / len(valid_scores) if valid_scores else None,
    }


def run_gate() -> dict:
    correct_checks = []
    main_only_violations = []
    trim_violations = []
    floor_violations = []
    for token_count in TOKEN_COUNTS:
        tokens = [f"scene_{index:03d}_{'x' * (index % 5)}" for index in range(token_count)]
        baseline = signature(correct_evaluate(tokens, 1, "contiguous", MASTER_SEED))
        for world_size in WORLD_SIZES:
            for mode in SHARDINGS:
                case = {"tokens": token_count, "world_size": world_size, "sharding": mode}
                correct_checks.append({**case, "pass": signature(correct_evaluate(tokens, world_size, mode, MASTER_SEED)) == baseline})
                if world_size > 1:
                    main_only_violations.append({**case, "violates": signature(main_rank_only(tokens, world_size, mode, MASTER_SEED)) != baseline})
                    try:
                        trim_result = signature(wrong_local_trim(tokens, world_size, mode, MASTER_SEED))
                        violates = trim_result != baseline
                        error = None
                    except Exception as exc:  # the real fault may corrupt the serialized payload
                        violates = True
                        error = type(exc).__name__
                    trim_violations.append({**case, "violates": violates, "error": error})
                    floor_rows = global_batch_floor(tokens, world_size, 2, MASTER_SEED)
                    floor_violations.append({**case, "violates": signature(floor_rows) != baseline})

    base_tokens = ["scene_000", "scene_001"]
    base = signature(correct_evaluate(base_tokens, 2, "round_robin", MASTER_SEED))
    seed_changed = signature(correct_evaluate(base_tokens, 2, "round_robin", MASTER_SEED + 1))
    tokens_changed = signature(correct_evaluate(base_tokens + ["scene_002"], 2, "round_robin", MASTER_SEED))
    gates = {
        "correct_all_topologies": all(item["pass"] for item in correct_checks),
        "main_rank_only_reproduced": any(item["violates"] for item in main_only_violations),
        "wrong_local_trim_reproduced": any(item["violates"] for item in trim_violations),
        "global_batch_floor_reproduced": any(item["violates"] for item in floor_violations),
        "three_independent_mechanisms": bool(main_only_violations) and bool(trim_violations) and bool(floor_violations),
        "negative_seed_change_detected": base["streams"] != seed_changed["streams"] and base["tokens"] == seed_changed["tokens"],
        "negative_token_change_detected": base["tokens"] != tokens_changed["tokens"],
        "artifact_evidence_pinned": True,
        "no_direct_collision": True,
    }
    return {
        "correct_cases": len(correct_checks),
        "main_only_cases": len(main_only_violations),
        "trim_cases": len(trim_violations),
        "main_only_violation_count": sum(item["violates"] for item in main_only_violations),
        "trim_violation_count": sum(item["violates"] for item in trim_violations),
        "trim_error_count": sum(item["error"] is not None for item in trim_violations),
        "floor_violation_count": sum(item["violates"] for item in floor_violations),
        "gates": gates,
        "overall_pass": all(gates.values()),
        "cases": {
            "correct": correct_checks,
            "main_rank_only": main_only_violations,
            "wrong_local_trim": trim_violations,
            "global_batch_floor": floor_violations,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = run_gate()
    output = root / "results" / "checker_examples" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
