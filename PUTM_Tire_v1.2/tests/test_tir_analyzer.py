import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mf51_model import MF51Model
from tir_parser import parse_tir_content, parse_tir_file, load_tir_to_model


class TirParserTests(unittest.TestCase):
    def setUp(self):
        self.base_model = MF51Model()
        self.base_model.set("PDY1", 1.72)
        self.base_model.set("PDX1", 1.85)
        self.base_model.set("PCY1", 1.35)
        self.base_model.set("FNOMIN", 1100.0)
        self.tir_text = self.base_model.to_tir_content("Hoosier_Test_Tyre")

    def test_parse_content(self):
        params, meta = parse_tir_content(self.tir_text, fallback_name="Default")
        self.assertEqual(meta.get("FILE_TYPE"), "tir")
        self.assertAlmostEqual(params.get("PDY1"), 1.72)
        self.assertAlmostEqual(params.get("PDX1"), 1.85)
        self.assertAlmostEqual(params.get("PCY1"), 1.35)
        self.assertAlmostEqual(params.get("FNOMIN"), 1100.0)

    def test_load_tir_to_model(self):
        temp_path = "temp_test_tyre.tir"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(self.tir_text)

            model, meta = load_tir_to_model(temp_path)
            self.assertIsInstance(model, MF51Model)
            self.assertAlmostEqual(model.get("PDY1"), 1.72)
            self.assertAlmostEqual(model.get("FNOMIN"), 1100.0)

            # Check calculations under operating conditions
            alpha_rad = np.deg2rad(np.linspace(-10, 10, 101))
            fy, meta_fy = model.evaluate_fy(alpha_rad, fz=1100.0, gamma_rad=0.0)
            self.assertAlmostEqual(meta_fy["Dy"], 1.72 * 1100.0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
