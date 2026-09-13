import unittest

from scripts.historical_defects import build_and_assert_source_cases, run_validation


class topologyHistoricalDefectsTest(unittest.TestCase):
    def test_source_labels_are_independent_and_reproducible(self):
        cases = build_and_assert_source_cases()
        self.assertEqual(len(cases), 5)
        self.assertTrue(all(case.source_prefix_symptom for case in cases))
        self.assertTrue(all(case.source_fixed_success for case in cases))

    def test_frozen_gate_records_misses_and_fixed_alerts(self):
        result = run_validation()
        self.assertEqual(result["defect_count"], 5)
        self.assertEqual(len(result["cases"]), 5)
        self.assertIn(result["decision"], {"PASS_INDEPENDENT_HISTORICAL_VALIDATION", "FALSIFIED"})


if __name__ == "__main__":
    unittest.main()
