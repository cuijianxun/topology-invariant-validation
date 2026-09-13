import csv
import tempfile
import unittest
from pathlib import Path

from scripts.repository_audit import audit_gate


ROOT = Path(__file__).resolve().parents[1]


class RepositoryAuditGateTest(unittest.TestCase):
    def test_frozen_audit_passes(self):
        result = audit_gate(
            ROOT / "data/repository_registry.csv",
            ROOT / "data/repository_audit.csv",
        )
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["registry_candidates"], 16)
        self.assertEqual(result["eligible_repositories"], 13)
        self.assertEqual(result["positive_static_signals"], 7)
        self.assertEqual(result["newly_inspected_positive_signals"], 4)
        self.assertLessEqual(result["unknown_semantic_cell_rate"], 0.20)

    def test_registry_order_drift_fails(self):
        audit_path = ROOT / "data/repository_audit.csv"
        with audit_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        rows[0], rows[1] = rows[1], rows[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            changed = Path(tmpdir) / "audit.csv"
            with changed.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            result = audit_gate(ROOT / "data/repository_registry.csv", changed)
        self.assertFalse(result["passed"])
        self.assertIn("audit rows do not match frozen registry order", result["errors"])


if __name__ == "__main__":
    unittest.main()
