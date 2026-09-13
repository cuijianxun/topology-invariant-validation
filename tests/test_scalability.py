import unittest

from scripts.scalability import percentile


class ScalabilityHelperTest(unittest.TestCase):
    def test_percentile_is_bounded(self):
        values = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(percentile(values, 0.25), 1.0)
        self.assertEqual(percentile(values, 0.95), 3.0)


if __name__ == "__main__":
    unittest.main()
