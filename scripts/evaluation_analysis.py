"""Statistical analysis for the frozen topology mutation experiment."""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mutations", ROOT / "scripts" / "mutation_benchmark.py")
MUT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MUT)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def mcnemar_exact(contract_names: set[str], baseline_names: set[str]) -> dict:
    contract_only = len(contract_names - baseline_names)
    baseline_only = len(baseline_names - contract_names)
    discordant = contract_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(0, min(contract_only, baseline_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {"contract_only": contract_only, "baseline_only": baseline_only, "discordant": discordant, "two_sided_p": p_value}


def diagnose(candidate: dict, baseline: dict) -> str | None:
    required = {"token", "stream", "failed", "score", "trace"}
    for item in candidate.get("rows", []):
        if not required.issubset(item) or not isinstance(item.get("failed"), bool):
            return "FAILURE_ACCOUNTING"
    expected_rows = {item["token"]: item for item in baseline["rows"]}
    observed_rows = {item["token"]: item for item in candidate["rows"]}
    expected_tokens = set(expected_rows)
    observed_tokens = set(observed_rows)
    if expected_tokens != observed_tokens or len(candidate["rows"]) != len(baseline["rows"]):
        missing = expected_tokens - observed_tokens
        expected_failed = {token for token, item in expected_rows.items() if item["failed"]}
        if missing and missing.issubset(expected_failed) and not (observed_tokens - expected_tokens):
            return "FAILURE_ACCOUNTING"
        return "SCENARIO_SET"
    if any(observed_rows[token]["stream"] != expected_rows[token]["stream"] for token in expected_tokens):
        return "RNG_IDENTITY"
    if any(observed_rows[token]["failed"] != expected_rows[token]["failed"] for token in expected_tokens):
        return "FAILURE_ACCOUNTING"
    for token in expected_tokens:
        if not expected_rows[token]["failed"] and not math.isclose(
            observed_rows[token]["score"], expected_rows[token]["score"], rel_tol=0, abs_tol=MUT.TOLERANCE
        ):
            return "REDUCTION"
    recomputed = MUT.valid_mean(candidate["rows"])
    if not math.isclose(candidate["reported_mean"], recomputed, rel_tol=0, abs_tol=MUT.TOLERANCE):
        return "REDUCTION"
    if not math.isclose(candidate["reported_mean"], baseline["reported_mean"], rel_tol=0, abs_tol=MUT.TOLERANCE):
        return "REDUCTION"
    return None


def expected_family(name: str) -> str:
    prefix = name.split("_", 1)[0]
    return {
        "scenario": "SCENARIO_SET",
        "rng": "RNG_IDENTITY",
        "failure": "FAILURE_ACCOUNTING",
        "aggregate": "REDUCTION",
    }[prefix]


def analyze() -> dict:
    summary_path = ROOT / "results" / "checker_examples" / "mutation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = summary["results"]["contract"]
    contract_names = set(contract["detected_names"])
    comparisons = {}
    for baseline in ("schema_only", "scalar_golden", "generic_equality"):
        baseline_result = summary["results"][baseline]
        comparisons[baseline] = {
            "absolute_fdr_gain": contract["fault_detection_rate"] - baseline_result["fault_detection_rate"],
            "mcnemar_exact": mcnemar_exact(contract_names, set(baseline_result["detected_names"])),
        }

    base = MUT.make_base()
    diagnostic_rows = []
    family_counts: dict[str, list[bool]] = defaultdict(list)
    for name, candidate in MUT.mutations(base):
        expected = expected_family(name)
        observed = diagnose(candidate, base)
        correct = expected == observed
        diagnostic_rows.append({"mutation": name, "expected": expected, "observed": observed, "correct": correct})
        family_counts[expected].append(correct)

    family_diagnosis = {
        family: {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
        for family, values in family_counts.items()
    }
    return {
        "contract_fdr": contract["fault_detection_rate"],
        "contract_fdr_wilson95": wilson(contract["detected"], contract["total_mutants"]),
        "contract_false_alarm_rate": contract["false_alarms"] / contract["allowed_variations"],
        "contract_false_alarm_wilson95": wilson(contract["false_alarms"], contract["allowed_variations"]),
        "comparisons": comparisons,
        "diagnosis": {
            "correct": sum(row["correct"] for row in diagnostic_rows),
            "total": len(diagnostic_rows),
            "accuracy": sum(row["correct"] for row in diagnostic_rows) / len(diagnostic_rows),
            "family": family_diagnosis,
            "rows": diagnostic_rows,
        },
    }


def main() -> None:
    output = analyze()
    path = ROOT / "results" / "evaluation" / "formal_analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    compact = dict(output)
    compact["diagnosis"] = {key: value for key, value in output["diagnosis"].items() if key != "rows"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
