import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mutations", ROOT / "scripts" / "mutation_benchmark.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class topologyMutationBenchmarkTest(unittest.TestCase):
    def test_revision_gate(self):
        summary = MODULE.run_benchmark()
        self.assertEqual(summary["mutants"], 48)
        self.assertEqual(summary["allowed_variations"], 24)
        self.assertTrue(summary["overall_pass"])
        self.assertEqual(summary["results"]["contract"]["false_alarms"], 0)


if __name__ == "__main__":
    unittest.main()
