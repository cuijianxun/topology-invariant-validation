"""Independent historical-defect validation for the frozen topology checker.

Discovery labels and pre/post predicates are transcribed from public maintainer
records before the topology checker is imported. Adapters are patch-extracted CPU
fixtures, not executions of the original packages.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "historical_defect_registry.csv"


@dataclass(frozen=True)
class SourceCase:
    historical_id: str
    baseline_tokens: tuple[str, ...]
    prefix_tokens: tuple[str, ...]
    fixed_tokens: tuple[str, ...]
    source_prefix_symptom: bool
    source_fixed_success: bool
    source_details: dict[str, object]


def _h1_gather_object() -> SourceCase:
    baseline = ("obj-rank-0", "obj-rank-1")
    prefix = ("obj-rank-0",)
    fixed = baseline
    return SourceCase(
        "HIST-001", baseline, prefix, fixed,
        source_prefix_symptom=prefix == ("obj-rank-0",),
        source_fixed_success=fixed == baseline,
        source_details={"prefix_gathered_ranks": [0], "fixed_gathered_ranks": [0, 1]},
    )


def _sharded_permutation(seed: int, rank: int, size: int = 22, world_size: int = 2) -> list[int]:
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices[rank::world_size]


def _h2_sampler_generator() -> SourceCase:
    baseline = tuple(f"sample-{i:02d}" for i in range(22))
    prefix_indices = _sharded_permutation(11, 0) + _sharded_permutation(29, 1)
    fixed_indices = _sharded_permutation(11, 0) + _sharded_permutation(11, 1)
    prefix = tuple(f"sample-{i:02d}" for i in prefix_indices)
    fixed = tuple(f"sample-{i:02d}" for i in fixed_indices)
    return SourceCase(
        "HIST-002", baseline, prefix, fixed,
        source_prefix_symptom=set(prefix) != set(baseline),
        source_fixed_success=set(fixed) == set(baseline) and len(fixed) == len(baseline),
        source_details={
            "prefix_unique": len(set(prefix)),
            "fixed_unique": len(set(fixed)),
            "expected_unique": len(baseline),
        },
    )


def _h3_dispatcher_wait() -> SourceCase:
    baseline = tuple(f"pipe-{i}" for i in range(6))
    prefix = baseline[::2]
    fixed = baseline
    return SourceCase(
        "HIST-003", baseline, prefix, fixed,
        source_prefix_symptom=len(prefix) < len(baseline),
        source_fixed_success=fixed == baseline,
        source_details={
            "prefix_completed_ranks": [0],
            "fixed_completed_ranks": [0, 1],
            "prefix_outcome": "timeout/incomplete loop",
            "fixed_outcome": "completed loop",
        },
    )


def _h4_implicit_padding() -> SourceCase:
    baseline = ("bond", "molecule", "element")
    prefix = ("bond", "molecule", "element", "element")
    fixed = baseline
    return SourceCase(
        "HIST-004", baseline, prefix, fixed,
        source_prefix_symptom=prefix.count("element") == 2,
        source_fixed_success=fixed.count("element") == 1 and fixed == baseline,
        source_details={"prefix_empty_rank_slice": ["element"], "fixed_empty_rank_slice": []},
    )


def _h5_distributed_sampler_drop_last() -> SourceCase:
    baseline = tuple(f"validation-{i}" for i in range(5))
    prefix = ("validation-0", "validation-2", "validation-4", "validation-1", "validation-3", "validation-0")
    fixed = ("validation-0", "validation-2", "validation-1", "validation-3")
    return SourceCase(
        "HIST-005", baseline, prefix, fixed,
        source_prefix_symptom=len(prefix) != len(set(prefix)),
        source_fixed_success=len(fixed) == len(set(fixed)),
        source_details={
            "prefix_duplicate_count": len(prefix) - len(set(prefix)),
            "fixed_duplicate_count": len(fixed) - len(set(fixed)),
            "fixed_dropped_tail": "validation-4",
        },
    )


SOURCE_BUILDERS = (
    _h1_gather_object,
    _h2_sampler_generator,
    _h3_dispatcher_wait,
    _h4_implicit_padding,
    _h5_distributed_sampler_drop_last,
)


def build_and_assert_source_cases() -> list[SourceCase]:
    cases = [builder() for builder in SOURCE_BUILDERS]
    assert len({case.historical_id for case in cases}) == len(cases)
    assert all(case.source_prefix_symptom for case in cases)
    assert all(case.source_fixed_success for case in cases)
    return cases


def _load_frozen_checker():
    spec = importlib.util.spec_from_file_location("frozen_gate", ROOT / "scripts" / "topology_checker.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stable_stream(token: str) -> int:
    return int.from_bytes(hashlib.sha256(("historical:" + token).encode()).digest()[:8], "big")


def _rows(tokens: tuple[str, ...]) -> list[dict]:
    rows = []
    for token in tokens:
        stream = _stable_stream(token)
        rows.append({"token": token, "stream": stream, "failed": False, "score": random.Random(stream).random(), "trace": token})
    return rows


def _posthoc_relation(reference: dict, observed: dict) -> str | None:
    if observed["tokens"] != reference["tokens"]:
        return "SCENARIO_SET"
    if observed["streams"] != reference["streams"]:
        return "RNG_IDENTITY"
    if observed["failed"] != reference["failed"]:
        return "FAILURE_ACCOUNTING"
    if observed["mean"] != reference["mean"]:
        return "REDUCTION"
    return None


def run_validation() -> dict[str, object]:
    # Source assertions are intentionally sealed before importing topology.
    cases = build_and_assert_source_cases()
    gate = _load_frozen_checker()
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        registry_rows = list(csv.DictReader(handle))
    assert [row["historical_id"] for row in registry_rows] == [case.historical_id for case in cases]

    results = []
    for case, registry in zip(cases, registry_rows):
        reference = gate.signature(_rows(case.baseline_tokens))
        prefix_signature = gate.signature(_rows(case.prefix_tokens))
        fixed_signature = gate.signature(_rows(case.fixed_tokens))
        prefix_relation = _posthoc_relation(reference, prefix_signature)
        fixed_relation = _posthoc_relation(reference, fixed_signature)
        results.append({
            "historical_id": case.historical_id,
            "repository": registry["repository"],
            "public_record": registry["public_record"],
            "immutable_fix_or_revision": registry["immutable_fix_or_revision"],
            "maintainer_title": registry["maintainer_title"],
            "source_symptom": registry["source_symptom"],
            "source_success_predicate": registry["source_success_predicate"],
            "source_prefix_assertion_pass": case.source_prefix_symptom,
            "source_fixed_assertion_pass": case.source_fixed_success,
            "source_details": case.source_details,
            "prefix_detected": prefix_relation is not None,
            "prefix_relation_posthoc": prefix_relation,
            "fixed_alert": fixed_relation is not None,
            "fixed_relation_posthoc": fixed_relation,
        })

    defect_count = len(results)
    repository_count = len({result["repository"] for result in results})
    detections = sum(result["prefix_detected"] for result in results)
    fixed_alerts = sum(result["fixed_alert"] for result in results)
    gates = {
        "at_least_five_historical_defects": defect_count >= 5,
        "at_least_two_independent_repositories_or_frameworks": repository_count >= 2,
        "all_source_pre_post_assertions_pass": all(
            result["source_prefix_assertion_pass"] and result["source_fixed_assertion_pass"] for result in results
        ),
        "detection_rate_at_least_70pct": detections / defect_count >= 0.70,
        "no_alert_on_source_fixed_versions": fixed_alerts == 0,
    }
    if not gates["at_least_five_historical_defects"]:
        decision = "INSUFFICIENT_PUBLIC_HISTORY"
    elif all(gates.values()):
        decision = "PASS_INDEPENDENT_HISTORICAL_VALIDATION"
    else:
        decision = "FALSIFIED"
    return {
        "schema_version": 1,
        "protocol": "experiments/historical_defect_protocol.md",
        "evidence_level": "PUBLIC_HISTORY_PATCH_EXTRACTED_CPU_FIXTURES_NOT_ORIGINAL_PACKAGE_EXECUTION",
        "defect_count": defect_count,
        "repository_or_framework_count": repository_count,
        "detections": detections,
        "detection_rate": detections / defect_count,
        "fixed_alerts": fixed_alerts,
        "gates": gates,
        "decision": decision,
        "cases": results,
    }


def main() -> None:
    summary = run_validation()
    output = ROOT / "results" / "historical" / "historical_defect_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
