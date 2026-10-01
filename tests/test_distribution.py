import unittest

import numpy as np

from distribution import make_histogram


class HistogramTests(unittest.TestCase):
    def test_counts_discrete_values_and_missing_samples(self):
        result = make_histogram([80, 80, 90, 90, 90, np.nan, np.inf])
        self.assertEqual(result.counts.tolist(), [2, 3])
        self.assertEqual((result.finite, result.total), (5, 7))

    def test_constant_channel_has_one_visible_bucket(self):
        result = make_histogram(np.full(200, 2.5))
        self.assertEqual(result.counts.tolist(), [200])
        self.assertLess(result.edges[0], 2.5)
        self.assertGreater(result.edges[-1], 2.5)

    def test_focus_reports_both_tails_and_caps_bin_count(self):
        values = np.concatenate(([-1000.0], np.linspace(0, 10, 1000), [1000.0]))
        result = make_histogram(values, focus=True)
        self.assertGreater(result.below, 0)
        self.assertGreater(result.above, 0)
        self.assertLessEqual(len(result.counts), 48)
        self.assertEqual(result.counts.sum() + result.below + result.above, result.finite)


if __name__ == "__main__":
    unittest.main()
