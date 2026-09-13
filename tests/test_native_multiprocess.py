import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native", ROOT / "scripts" / "native_multiprocess.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NativeMultiprocessHelpersTest(unittest.TestCase):
    def test_worker_preserves_token_keyed_identity(self):
        result = MODULE.worker((["a", "b"], MODULE.GATE.MASTER_SEED))
        self.assertEqual([row["token"] for row in result["rows"]], ["a", "b"])
        self.assertEqual(result["rows"][0]["stream"], MODULE.GATE.keyed_randomness(MODULE.GATE.MASTER_SEED, "a"))


if __name__ == "__main__":
    unittest.main()
