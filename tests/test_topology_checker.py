import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("topology_checker", ROOT / "scripts" / "topology_checker.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class topologyTopologyGateTest(unittest.TestCase):
    def test_keyed_randomness_is_order_independent(self):
        tokens = ["a", "b", "c"]
        forward = MODULE.signature(MODULE.correct_evaluate(tokens, 2, "round_robin", MODULE.MASTER_SEED))
        reverse = MODULE.signature(MODULE.correct_evaluate(list(reversed(tokens)), 2, "contiguous", MODULE.MASTER_SEED))
        self.assertEqual(forward, reverse)

    def test_main_rank_only_loses_scenarios(self):
        tokens = [f"s{i}" for i in range(7)]
        self.assertNotEqual(
            MODULE.signature(MODULE.main_rank_only(tokens, 3, "contiguous", MODULE.MASTER_SEED)),
            MODULE.signature(MODULE.correct_evaluate(tokens, 1, "contiguous", MODULE.MASTER_SEED)),
        )

    def test_gate_passes(self):
        summary = MODULE.run_gate()
        self.assertTrue(summary["overall_pass"])
        self.assertEqual(summary["correct_cases"], 24)
        self.assertGreater(summary["main_only_violation_count"], 0)
        self.assertGreater(summary["trim_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
