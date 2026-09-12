import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.selective import risk_at_coverage, risk_coverage_curve


class SelectiveMetricsTest(unittest.TestCase):
    def test_risk_at_coverage_uses_target_point_not_first_list_item(self):
        y_true = [0, 0, 0, 0, 0]
        y_pred = [0, 1, 1, 1, 1]
        reliability = [5, 4, 3, 2, 1]

        curve = risk_coverage_curve(y_true, y_pred, reliability, coverages=[1.0, 0.8])

        self.assertEqual(curve[0]["coverage"], 1.0)
        self.assertEqual(risk_at_coverage(curve, 0.8), 0.75)


if __name__ == "__main__":
    unittest.main()
