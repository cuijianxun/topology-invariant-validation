#!/usr/bin/env python3
"""Validate the frozen topology repository audit without network or third-party packages."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SEMANTIC_FIELDS = (
    "distributed_entry",
    "scenario_sharding",
    "rng_binding",
    "failure_semantics",
    "reduction",
    "topology_test",
)


def audit_gate(registry_path: Path, audit_path: Path) -> dict:
    with registry_path.open(encoding="utf-8", newline="") as handle:
        registry = list(csv.DictReader(handle))
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit = list(csv.DictReader(handle))

    registry_ids = [row["registry_id"] for row in registry]
    audit_ids = [row["registry_id"] for row in audit]
    errors: list[str] = []

    if len(registry) != 16:
        errors.append(f"expected 16 frozen candidates, found {len(registry)}")
    if len(set(registry_ids)) != len(registry_ids):
        errors.append("registry_id is not unique")
    if audit_ids != registry_ids:
        errors.append("audit rows do not match frozen registry order")

    eligible = [row for row in audit if row["eligibility"] == "ELIGIBLE"]
    ineligible = [row for row in audit if row["eligibility"].startswith("INELIGIBLE_")]
    positive = [row for row in eligible if row["topology_signal"].startswith("YES_")]
    new_positive = [row for row in positive if row["prior_code_exposure"] == "NONE_BEFORE_FREEZE"]
    prior_positive = [row for row in positive if row["prior_code_exposure"] == "POSITIVE_PATH_PREVIOUSLY_READ"]

    if len(eligible) < 8:
        errors.append(f"eligible repository gate failed: {len(eligible)} < 8")
    if len(eligible) + len(ineligible) != len(audit):
        errors.append("every audit row must be eligible or carry an ineligible reason")

    sha_pattern = re.compile(r"^[0-9a-f]{40}$")
    for row in audit:
        if not sha_pattern.fullmatch(row["commit_sha"]):
            errors.append(f"{row['registry_id']} has invalid commit SHA")
        if row["eligibility"] == "ELIGIBLE" and not row["entry_point"].strip():
            errors.append(f"{row['registry_id']} lacks an entry point")
        if row["topology_signal"].startswith("YES_") and "STATIC" not in row["evidence_level"]:
            errors.append(f"{row['registry_id']} positive signal lacks static evidence label")

    coded_cells = [row[field] for row in eligible for field in SEMANTIC_FIELDS]
    unknown_cells = [value for value in coded_cells if value == "UNKNOWN"]
    unknown_rate = len(unknown_cells) / len(coded_cells) if coded_cells else 1.0
    if unknown_rate > 0.20:
        errors.append(f"unknown semantic-cell rate {unknown_rate:.3f} exceeds 0.20")

    result = {
        "schema_version": 1,
        "registry_candidates": len(registry),
        "eligible_repositories": len(eligible),
        "ineligible_repositories": len(ineligible),
        "positive_static_signals": len(positive),
        "newly_inspected_positive_signals": len(new_positive),
        "previously_known_positive_signals": len(prior_positive),
        "unknown_semantic_cells": len(unknown_cells),
        "coded_semantic_cells": len(coded_cells),
        "unknown_semantic_cell_rate": unknown_rate,
        "eligibility_counts": dict(Counter(row["eligibility"] for row in audit)),
        "passed": not errors,
        "errors": errors,
        "reporting_constraint": "DESCRIPTIVE_MULTI_CASE_AUDIT_NOT_PREVALENCE",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("data/repository_registry.csv"))
    parser.add_argument("--audit", type=Path, default=Path("data/repository_audit.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation/repository_audit_gate.json"))
    args = parser.parse_args()

    result = audit_gate(args.registry, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
