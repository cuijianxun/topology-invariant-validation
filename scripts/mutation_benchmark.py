"""Fault-injection comparison for the topology topology contract."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-9
SPEC = importlib.util.spec_from_file_location("gate", ROOT / "scripts" / "topology_checker.py")
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def valid_mean(rows: list[dict]) -> float | None:
    values = [item["score"] for item in rows if not item["failed"] and item["score"] is not None]
    return sum(values) / len(values) if values else None


def artifact(rows: list[dict], reported_mean: float | None = None) -> dict:
    return {"rows": rows, "reported_mean": valid_mean(rows) if reported_mean is None else reported_mean}


def schema_detect(candidate: dict, baseline: dict) -> bool:
    if not isinstance(candidate.get("rows"), list):
        return True
    required = {"token", "stream", "failed", "score", "trace"}
    for item in candidate["rows"]:
        if not required.issubset(item):
            return True
        if not isinstance(item["token"], str) or not isinstance(item["stream"], int) or not isinstance(item["failed"], bool):
            return True
    return False


def scalar_golden_detect(candidate: dict, baseline: dict) -> bool:
    return not math.isclose(candidate["reported_mean"], baseline["reported_mean"], rel_tol=0, abs_tol=TOLERANCE)


def generic_equality_detect(candidate: dict, baseline: dict) -> bool:
    return len(candidate["rows"]) != len(baseline["rows"]) or scalar_golden_detect(candidate, baseline)


def contract_detect(candidate: dict, baseline: dict) -> bool:
    if schema_detect(candidate, baseline):
        return True
    expected = GATE.signature(baseline["rows"])
    observed = GATE.signature(candidate["rows"])
    if observed["tokens"] != expected["tokens"]:
        return True
    if observed["streams"] != expected["streams"]:
        return True
    if observed["failed"] != expected["failed"]:
        return True
    expected_scores = {item["token"]: item["score"] for item in baseline["rows"] if not item["failed"]}
    observed_scores = {item["token"]: item["score"] for item in candidate["rows"] if not item["failed"]}
    if set(expected_scores) != set(observed_scores):
        return True
    if any(not math.isclose(observed_scores[token], expected_scores[token], rel_tol=0, abs_tol=TOLERANCE) for token in expected_scores):
        return True
    recomputed = valid_mean(candidate["rows"])
    return not math.isclose(candidate["reported_mean"], recomputed, rel_tol=0, abs_tol=TOLERANCE) or not math.isclose(
        candidate["reported_mean"], baseline["reported_mean"], rel_tol=0, abs_tol=TOLERANCE
    )


def make_base() -> dict:
    rows = [GATE.row(f"scene_{i:03d}", GATE.MASTER_SEED, failed=i in {5, 13}) for i in range(20)]
    return artifact(rows)


def mutations(base: dict) -> list[tuple[str, dict]]:
    output: list[tuple[str, dict]] = []

    def add(name: str, rows: list[dict], mean: float | None = None) -> None:
        output.append((name, artifact(rows, mean)))

    rows = base["rows"]
    add("scenario_drop_first", copy.deepcopy(rows[1:]))
    add("scenario_drop_last", copy.deepcopy(rows[:-1]))
    add("scenario_duplicate_first", copy.deepcopy(rows + [rows[0]]))
    add("scenario_replace_first", copy.deepcopy([rows[1]] + rows[1:]))
    add("scenario_even_only", copy.deepcopy(rows[::2]))
    add("scenario_global_batch_floor", copy.deepcopy(rows[:16]))
    add("scenario_drop_middle", copy.deepcopy(rows[:9] + rows[10:]))
    add("scenario_duplicate_last", copy.deepcopy(rows + [rows[-1]]))
    add("scenario_replace_last", copy.deepcopy(rows[:-1] + [rows[0]]))
    add("scenario_truncate_18", copy.deepcopy(rows[:18]))
    added = copy.deepcopy(rows)
    added.append(GATE.row("scene_new", GATE.MASTER_SEED))
    add("scenario_add_new", added)
    changed = copy.deepcopy(rows)
    changed[8]["token"] = "scene_alias"
    add("scenario_alias_one", changed)

    for name, transform in (
        ("rng_rotate", lambda values: values[1:] + values[:1]),
        ("rng_reverse", lambda values: list(reversed(values))),
    ):
        changed = copy.deepcopy(rows)
        values = transform([item["stream"] for item in changed])
        for item, value in zip(changed, values):
            item["stream"] = value
        add(name, changed)
    changed = copy.deepcopy(rows)
    for item in changed:
        item["stream"] = 42
    add("rng_same_all", changed)
    changed = copy.deepcopy(rows)
    changed[0]["stream"] += 1
    add("rng_one_changed", changed)
    changed = copy.deepcopy(rows)
    for index, item in enumerate(changed):
        item["stream"] = GATE.keyed_randomness(GATE.MASTER_SEED + index, item["token"])
    add("rng_rank_local", changed)
    changed = copy.deepcopy(rows)
    for item in changed:
        item["stream"] ^= 0xFFFF
    add("rng_xor_all", changed)
    changed = copy.deepcopy(rows)
    changed[0]["stream"], changed[1]["stream"] = changed[1]["stream"], changed[0]["stream"]
    add("rng_swap_pair", changed)
    changed = copy.deepcopy(rows)
    for index, item in enumerate(changed):
        item["stream"] = GATE.keyed_randomness(index % 3, item["token"])
    add("rng_rank_hash", changed)
    changed = copy.deepcopy(rows)
    for item in changed:
        item["stream"] = GATE.keyed_randomness(GATE.MASTER_SEED + 1, item["token"])
    add("rng_seed_off_by_one", changed)
    changed = copy.deepcopy(rows)
    for item in changed:
        item["stream"] &= 0xFFFF
    add("rng_truncated", changed)
    changed = copy.deepcopy(rows)
    for index, item in enumerate(changed):
        item["stream"] = index
    add("rng_position_based", changed)
    changed = copy.deepcopy(rows)
    for index in range(0, len(changed), 2):
        changed[index]["stream"] = 0
    add("rng_zero_even", changed)

    add("failure_drop_failed", copy.deepcopy([item for item in rows if not item["failed"]]))
    changed = copy.deepcopy(rows)
    changed[5]["failed"] = False
    changed[5]["score"] = 0.0
    add("failure_clear_one", changed)
    changed = copy.deepcopy(rows)
    changed[0]["failed"] = True
    changed[0]["score"] = None
    add("failure_add_one", changed)
    changed = copy.deepcopy(rows)
    changed[5]["failed"], changed[0]["failed"] = False, True
    changed[5]["score"], changed[0]["score"] = 0.0, None
    add("failure_swap_identity", changed)
    changed = copy.deepcopy(rows)
    for item in changed:
        item["failed"] = False
        if item["score"] is None:
            item["score"] = 0.0
    add("failure_clear_all", changed)
    changed = copy.deepcopy(rows)
    del changed[0]["failed"]
    add("failure_schema_loss", changed, base["reported_mean"])
    add("failure_drop_first_failed", copy.deepcopy([item for index, item in enumerate(rows) if index != 5]))
    add("failure_drop_second_failed", copy.deepcopy([item for index, item in enumerate(rows) if index != 13]))
    changed = copy.deepcopy(rows)
    changed[13]["failed"] = False
    changed[13]["score"] = 0.0
    add("failure_clear_second", changed)
    changed = copy.deepcopy(rows)
    changed[1]["failed"] = True
    changed[1]["score"] = None
    add("failure_add_second", changed)
    changed = copy.deepcopy(rows)
    changed[5]["failed"] = None
    add("failure_none_status", changed, base["reported_mean"])
    changed = copy.deepcopy(rows)
    changed[13]["failed"] = "failed"
    add("failure_string_status", changed, base["reported_mean"])

    add("aggregate_stale", copy.deepcopy(rows), base["reported_mean"] + 0.01)
    add("aggregate_failed_as_zero", copy.deepcopy(rows), sum((item["score"] or 0.0) for item in rows) / len(rows))
    add("aggregate_rank0_only", copy.deepcopy(rows), valid_mean(rows[:10]))
    add("aggregate_unweighted_shards", copy.deepcopy(rows), (valid_mean(rows[:7]) + valid_mean(rows[7:])) / 2)
    valid_values = [item["score"] for item in rows if not item["failed"]]
    add("aggregate_drop_max", copy.deepcopy(rows), (sum(valid_values) - max(valid_values)) / (len(valid_values) - 1))
    add("aggregate_rounded", copy.deepcopy(rows), round(base["reported_mean"], 2))
    add("aggregate_median", copy.deepcopy(rows), statistics.median(valid_values))
    add("aggregate_min", copy.deepcopy(rows), min(valid_values))
    add("aggregate_max", copy.deepcopy(rows), max(valid_values))
    add("aggregate_first_half", copy.deepcopy(rows), valid_mean(rows[:10]))
    changed = copy.deepcopy(rows)
    valid_positions = [index for index, item in enumerate(changed) if not item["failed"]]
    scores = [changed[index]["score"] for index in valid_positions]
    for index, score in zip(valid_positions, scores[1:] + scores[:1]):
        changed[index]["score"] = score
    add("aggregate_score_association_rotate", changed, base["reported_mean"])
    add("aggregate_epsilon_outside", copy.deepcopy(rows), base["reported_mean"] + 1e-6)
    return output


def allowed_variations(base: dict) -> list[tuple[str, dict]]:
    variants = []
    rows = base["rows"]
    variants.append(("reverse_order", artifact(copy.deepcopy(list(reversed(rows))))))
    shuffled = copy.deepcopy(rows)
    random.Random(7).shuffle(shuffled)
    variants.append(("shuffled_order", artifact(shuffled)))
    for index in range(6):
        changed = copy.deepcopy(rows)
        for item in changed:
            item["trace"] = item["trace"] + ("meta" * index)
        variants.append((f"trace_metadata_{index}", artifact(changed)))
    for index in range(8):
        changed = copy.deepcopy(rows)
        for rank, item in enumerate(changed):
            item["rank"] = (rank + index) % 4
            item["timing_ms"] = rank + index / 10
        variants.append((f"rank_metadata_{index}", artifact(changed)))
    for index in range(8):
        changed = copy.deepcopy(rows)
        delta = (index + 1) * TOLERANCE / 20
        for item in changed:
            if not item["failed"]:
                item["score"] += delta
        variants.append((f"float_within_tolerance_{index}", artifact(changed)))
    return variants


def run_benchmark() -> dict:
    base = make_base()
    detectors = {
        "schema_only": schema_detect,
        "scalar_golden": scalar_golden_detect,
        "generic_equality": generic_equality_detect,
        "contract": contract_detect,
    }
    mutant_rows = mutations(base)
    allowed_rows = allowed_variations(base)
    results = {}
    for name, detector in detectors.items():
        detected = [mutation for mutation, candidate in mutant_rows if detector(candidate, base)]
        false_alarms = [variation for variation, candidate in allowed_rows if detector(candidate, base)]
        results[name] = {
            "detected": len(detected),
            "total_mutants": len(mutant_rows),
            "fault_detection_rate": len(detected) / len(mutant_rows),
            "false_alarms": len(false_alarms),
            "allowed_variations": len(allowed_rows),
            "detected_names": detected,
            "false_alarm_names": false_alarms,
        }
    gates = {
        "at_least_20_mutants": len(mutant_rows) >= 20,
        "four_fault_families": len(mutant_rows) == 48,
        "contract_detects_all": results["contract"]["detected"] == len(mutant_rows),
        "contract_zero_false_alarm": results["contract"]["false_alarms"] == 0,
        "beats_all_baselines": all(
            results["contract"]["detected"] > results[name]["detected"]
            for name in ("schema_only", "scalar_golden", "generic_equality")
        ),
    }
    return {"mutants": len(mutant_rows), "allowed_variations": len(allowed_rows), "tolerance": TOLERANCE, "results": results, "gates": gates, "overall_pass": all(gates.values())}


def main() -> None:
    summary = run_benchmark()
    output = ROOT / "results" / "checker_examples" / "mutation_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
