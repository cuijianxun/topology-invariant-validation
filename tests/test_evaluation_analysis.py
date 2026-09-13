import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analysis", ROOT / "scripts" / "evaluation_analysis.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FormalAnalysisTest(unittest.TestCase):
    def test_wilson_is_bounded(self):
        lower, upper = MODULE.wilson(0, 24)
        self.assertGreaterEqual(lower, 0)
        self.assertLessEqual(upper, 1)

    def test_mcnemar_direction(self):
        result = MODULE.mcnemar_exact({"a", "b"}, {"a"})
        self.assertEqual(result["contract_only"], 1)
        self.assertEqual(result["baseline_only"], 0)


if __name__ == "__main__":
    unittest.main()
