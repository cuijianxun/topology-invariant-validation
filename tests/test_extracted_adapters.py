import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adapters", ROOT / "scripts" / "extracted_adapters.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExtractedAdapterHelpersTest(unittest.TestCase):
    def test_worker_rows_are_token_keyed(self):
        result = MODULE._token_worker((0, ["a", "b"], MODULE.GATE.MASTER_SEED))
        self.assertEqual([row["token"] for row in result["rows"]], ["a", "b"])
        self.assertEqual(
            result["rows"][0]["stream"],
            MODULE.GATE.keyed_randomness(MODULE.GATE.MASTER_SEED, "a"),
        )

    def test_payload_size_is_rank_dependent(self):
        small = MODULE._payload_worker((0, ["a"], MODULE.GATE.MASTER_SEED))["blob"]
        large = MODULE._payload_worker((1, ["a"], MODULE.GATE.MASTER_SEED))["blob"]
        self.assertGreater(len(large), len(small))


if __name__ == "__main__":
    unittest.main()
