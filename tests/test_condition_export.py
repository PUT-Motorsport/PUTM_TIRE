import os
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from condition_export import (
    NO_DATA, ConditionSpec, center_bins, classify_frame, collect_condition_data, format_summary,
    parse_float, parse_number_list, parse_range_list, range_bins, write_csv,
)
from tire_model import RunConfig


def frame(fz, ia, p):
    n = len(fz)
    return pd.DataFrame({"FZ": fz, "IA": ia, "P_kPa": p, "SA": np.zeros(n), "FY": np.zeros(n), "__row__": np.arange(n)})


class BinningTests(unittest.TestCase):
    def test_default_spec_matches_requested_conditions(self):
        spec = ConditionSpec.default()
        self.assertEqual([b.label for b in spec.fz_bins], ["600", "800", "1000", "1200"])
        self.assertEqual([(b.low, b.high) for b in spec.fz_bins][0], (400.0, 800.0))
        self.assertEqual([b.label for b in spec.ia_bins], ["0", "-1", "-2", "-3"])
        self.assertEqual((spec.ia_bins[1].low, spec.ia_bins[1].high), (-1.5, -0.5))
        self.assertEqual([b.label for b in spec.p_bins], ["50-60", "65-75", "80-85", "94-100"])

    def test_sample_lands_in_correct_cell(self):
        data, counts = classify_frame(frame([1000.0], [-1.0], [70.0]), ConditionSpec.default())
        self.assertEqual(len(data), 1)
        self.assertEqual(tuple(data.loc[0, ["Fz_bin_N", "IA_bin_deg", "P_bin_kPa"]]), ("1000", "-1", "65-75"))
        self.assertEqual(counts[("1000", "-1", "65-75")], 1)

    def test_pressure_gaps_are_excluded(self):
        data, _ = classify_frame(frame([1000.0, 1000.0, 1000.0], [0.0] * 3, [62.0, 90.0, 101.0]), ConditionSpec.default())
        self.assertEqual(len(data), 0)

    def test_literal_windows_overlap_and_duplicate_samples(self):
        literal = replace(ConditionSpec.default(), nearest_only=False)
        data, _ = classify_frame(frame([900.0], [0.0], [55.0]), literal)
        self.assertEqual(sorted(data["Fz_bin_N"]), ["1000", "800"])           # 900 N: 800±200 i 1000±200
        data, _ = classify_frame(frame([1000.0], [0.0], [55.0]), literal)
        self.assertEqual(sorted(data["Fz_bin_N"]), ["1000", "1200", "800"])   # 1000 N: aż trzy okna

    def test_default_is_nearest_center_without_duplicates(self):
        self.assertTrue(ConditionSpec.default().nearest_only)
        data, _ = classify_frame(frame([850.0, 950.0, 1000.0], [0.0] * 3, [55.0] * 3), ConditionSpec.default())
        self.assertEqual(list(data["Fz_bin_N"]), ["800", "1000", "1000"])

    def test_nearest_only_still_respects_tolerance(self):
        data, _ = classify_frame(frame([1450.0, 1390.0], [0.0, 0.0], [55.0, 55.0]), ConditionSpec.default())
        self.assertEqual(list(data["Fz_bin_N"]), ["1200"])   # 1450 N jest 250 N od 1200 - poza ±200

    def test_camber_boundaries_do_not_duplicate_by_default(self):
        data, _ = classify_frame(frame([1000.0], [-0.5], [55.0]), ConditionSpec.default())
        self.assertEqual(len(data), 1)

    def test_outside_fz_window_dropped(self):
        data, _ = classify_frame(frame([300.0, 1500.0], [0.0, 0.0], [55.0, 55.0]), ConditionSpec.default())
        self.assertEqual(len(data), 0)

    def test_missing_pressure_channel_marks_no_data(self):
        data, counts = classify_frame(frame([1000.0], [0.0], [np.nan]), ConditionSpec.default())
        self.assertEqual(list(data["P_bin_kPa"]), [NO_DATA])

    def test_partially_missing_pressure_excludes_nan_rows(self):
        data, _ = classify_frame(frame([1000.0, 1000.0], [0.0, 0.0], [55.0, np.nan]), ConditionSpec.default())
        self.assertEqual(list(data["P_bin_kPa"]), ["50-60"])


class ParsingTests(unittest.TestCase):
    def test_lists_and_ranges(self):
        self.assertEqual(parse_number_list("0, -1, -2, -3"), [0, -1, -2, -3])
        self.assertEqual(parse_number_list("600 800;1000"), [600, 800, 1000])
        self.assertEqual(parse_range_list("50-60, 65–75 ; 80 - 85, 94-100"), [(50, 60), (65, 75), (80, 85), (94, 100)])
        self.assertEqual(parse_float("0,5"), 0.5)

    def test_bad_input_rejected(self):
        for bad in ("abc", "", "1;;x"):
            with self.assertRaises(ValueError):
                parse_number_list(bad)
        with self.assertRaises(ValueError):
            parse_range_list("50-60, xyz")
        with self.assertRaises(ValueError):
            parse_range_list("60")
        with self.assertRaises(ValueError):
            range_bins([(60, 50)])
        with self.assertRaises(ValueError):
            center_bins([1000], -1)


class MatIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        rng = np.random.default_rng(0)
        n = 600
        fz = rng.choice([-600.0, -800.0, -1000.0, -1200.0], n)         # TTC: Fz ujemne
        ia = rng.choice([0.0, -1.0, -2.0, -3.0], n) + rng.normal(0, 0.1, n)
        pressure = rng.choice([55.0, 70.0, 82.0, 97.0, 62.0], n)       # 62 kPa leży poza zakresami
        sa = np.linspace(-12, 12, n)
        self.path = os.path.join(self.tmp.name, "run1.mat")
        scipy.io.savemat(self.path, {"SA": sa, "FY": -50 * sa, "FZ": fz, "IA": ia, "P": pressure, "ET": np.arange(n, dtype=float)})
        self.nop_path = os.path.join(self.tmp.name, "run_noP.mat")
        scipy.io.savemat(self.nop_path, {"SA": sa, "FY": -50 * sa, "FZ": fz, "IA": ia})
        self.fz, self.pressure = np.abs(fz), pressure

    def run_cfg(self, path, test_type="Cornering"):
        return RunConfig(filepath=path, test_type=test_type, target_fz=1000, tol_fz=100, target_param=0, tol_param=0.25)

    def test_one_file_covers_all_conditions(self):
        result = collect_condition_data([self.run_cfg(self.path)], ConditionSpec.default())
        self.assertEqual(set(result.data["Fz_bin_N"]), {"600", "800", "1000", "1200"})
        self.assertEqual(set(result.data["IA_bin_deg"]), {"0", "-1", "-2", "-3"})
        self.assertEqual(set(result.data["P_bin_kPa"]), {"50-60", "65-75", "80-85", "94-100"})
        # surowe wartości zostają, Fz jest dodatnie, ciśnienie w kPa
        self.assertTrue((result.data["FZ"] > 0).all())
        self.assertIn("P_kPa", result.data.columns)
        self.assertEqual(result.rows_exported, result.rows_unique)   # domyślnie bez duplikatów
        self.assertEqual(int(result.summary["n_samples"].sum()), result.rows_exported)

    def test_pressure_62_is_outside_all_cells(self):
        result = collect_condition_data([self.run_cfg(self.path)], ConditionSpec.default())
        self.assertFalse(((result.data["P_kPa"] > 60) & (result.data["P_kPa"] < 65)).any())
        self.assertGreater(result.rows_outside, 0)

    def test_run_filters_do_not_limit_export(self):
        # run ma filtr Fz=1000±100, IA=0±0.25, ale eksport ma zwrócić wszystkie warunki
        result = collect_condition_data([self.run_cfg(self.path)], ConditionSpec.default())
        self.assertGreater(result.data["Fz_bin_N"].nunique(), 1)

    def test_duplicate_runs_of_same_file_counted_once(self):
        once = collect_condition_data([self.run_cfg(self.path)], ConditionSpec.default())
        twice = collect_condition_data([self.run_cfg(self.path), self.run_cfg(self.path)], ConditionSpec.default())
        self.assertEqual(once.rows_exported, twice.rows_exported)

    def test_missing_pressure_file_keeps_rows_and_warns(self):
        result = collect_condition_data([self.run_cfg(self.nop_path)], ConditionSpec.default())
        self.assertEqual(set(result.data["P_bin_kPa"]), {NO_DATA})
        self.assertTrue(any("missing pressure channel" in note for note in result.notes))

    def test_unit_conversion_psi(self):
        # plik z ciśnieniem w psi: 8 psi ≈ 55.2 kPa -> komórka 50-60 kPa (zakresy w spec są zawsze w kPa)
        n = 50
        path = os.path.join(self.tmp.name, "psi.mat")
        scipy.io.savemat(path, {"SA": np.linspace(-5, 5, n), "FY": np.zeros(n), "FZ": np.full(n, -1000.0),
                                "IA": np.zeros(n), "P": np.full(n, 8.0)})
        base = ConditionSpec.default()
        spec = replace(base, p_to_kpa=6.894757)
        data = collect_condition_data([self.run_cfg(path)], spec).data
        self.assertEqual(set(data["P_bin_kPa"]), {"50-60"})
        self.assertAlmostEqual(float(data["P_kPa"].iloc[0]), 55.158, places=2)
        # bez przelicznika 8 "kPa" leży poza wszystkimi zakresami
        self.assertTrue(collect_condition_data([self.run_cfg(path)], base).data.empty)

    def test_bad_file_reported_not_raised(self):
        bad = self.run_cfg(os.path.join(self.tmp.name, "nie_ma.mat"))
        result = collect_condition_data([bad, self.run_cfg(self.path)], ConditionSpec.default())
        self.assertEqual(result.files_used, 1)
        self.assertTrue(any("nie_ma.mat" in note for note in result.notes))

    def test_write_csv_round_trip_and_summary_text(self):
        result = collect_condition_data([self.run_cfg(self.path)], ConditionSpec.default())
        out = os.path.join(self.tmp.name, "macierz.csv")
        write_csv(out, result)
        loaded = pd.read_csv(out, dtype={"Fz_bin_N": str, "IA_bin_deg": str, "P_bin_kPa": str})
        self.assertEqual(len(loaded), result.rows_exported)
        self.assertEqual(list(loaded.columns[:6]),
                         ["Fz_bin_N", "IA_bin_deg", "P_bin_kPa", "__source__", "__type__", "__row__"])
        self.assertIn("Fz [N]", format_summary(result))

    def test_empty_result_not_written(self):
        base = ConditionSpec.default()
        empty = ConditionSpec(base.fz_bins, base.ia_bins, range_bins([(300, 310)]))
        result = collect_condition_data([self.run_cfg(self.path)], empty)
        with self.assertRaises(ValueError):
            write_csv(os.path.join(self.tmp.name, "x.csv"), result)


if __name__ == "__main__":
    unittest.main()
