import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from tire_model import RunConfig, TireDataManager


class TireDataManagerTests(unittest.TestCase):
    def test_drive_filter_preserves_operating_conditions_and_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.mat"
            savemat(
                path,
                {
                    "FX": np.array([100.0, 200.0, 300.0, 400.0]),
                    "FY": np.array([-500.0, -510.0, -520.0, -530.0]),
                    "FZ": np.array([-995.0, -1005.0, -1000.0, -1300.0]),
                    "SL": np.array([-0.1, 0.0, 0.1, 0.0]),
                    "SA": np.array([-3.0, -3.0, 0.0, -3.0]),
                    "IA": np.zeros(4),
                    "ET": np.arange(4, dtype=float),
                },
            )
            config = RunConfig(
                str(path), "Drive/Brake", 1000.0, 20.0, 0.0, 0.2,
                target_sa=-3.0, tol_sa=0.2, target_ia=0.0, tol_ia=0.1,
            )

            frame = TireDataManager.load_and_filter(config)
            raw = TireDataManager.load_unfiltered(config)

            self.assertEqual(frame["__row__"].tolist(), [0, 1])
            self.assertEqual(len(raw), 4)
            self.assertEqual(raw["FZ"].tolist(), [995.0, 1005.0, 1000.0, 1300.0])
            self.assertEqual(frame["SA"].tolist(), [-3.0, -3.0])
            self.assertEqual(frame["FZ"].tolist(), [995.0, 1005.0])
            self.assertEqual(frame["FZ_RAW"].tolist(), [-995.0, -1005.0])
            self.assertIn("ET", frame)

    def test_unknown_test_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.mat"
            savemat(path, {"FZ": np.array([-1000.0])})
            config = RunConfig(str(path), "Unknown", 1000.0, 20.0, 0.0, 0.1)
            with self.assertRaisesRegex(ValueError, "Nieobsługiwany"):
                TireDataManager.load_and_filter(config)

    def test_partial_condition_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Filtr SA"):
            RunConfig("unused.mat", "Drive/Brake", 1000.0, 20.0, 0.0, 0.2, target_sa=0.0)

    def test_cornering_can_be_loaded_without_optional_mz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corner.mat"
            savemat(
                path,
                {
                    "SA": np.array([-2.0, 0.0, 2.0]),
                    "FY": np.array([500.0, 0.0, -500.0]),
                    "FZ": np.array([-1000.0, -1000.0, -1000.0]),
                    "IA": np.zeros(3),
                },
            )
            frame = TireDataManager.load_and_filter(RunConfig(str(path), "Cornering", 1000.0, 20.0, 0.0, 0.1))
            self.assertEqual(len(frame), 3)
            self.assertTrue(frame["MZ"].isna().all())


if __name__ == "__main__":
    unittest.main()
