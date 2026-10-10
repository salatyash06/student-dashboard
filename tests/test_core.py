import importlib.util
import unittest
import pandas as pd


class InputContractTests(unittest.TestCase):
    def core(self):
        self.assertIsNotNone(importlib.util.find_spec("dashboard_core"),
                             "Validated prediction input preparation is missing")
        import dashboard_core
        return dashboard_core

    def test_original_encoding_and_feature_order(self):
        core = self.core()
        frame = core.prepare_input(dict(core.DEFAULT_PROFILE),
            ["Attendance", "Gender_Male", "Hours_Studied", "Physical_Activity"])
        pd.testing.assert_frame_equal(frame, pd.DataFrame({"Attendance": [80],
            "Gender_Male": [0], "Hours_Studied": [20], "Physical_Activity": [0]}))

    def test_invalid_values_never_reach_model(self):
        core = self.core()
        for field, value in [("Attendance", 101), ("Hours_Studied", -1),
                             ("Sleep_Hours", 25), ("Tutoring_Sessions", 1.5),
                             ("Previous_Scores", float("nan"))]:
            with self.subTest(field=field):
                profile = dict(core.DEFAULT_PROFILE, **{field: value})
                with self.assertRaises(ValueError):
                    core.prepare_input(profile, list(profile))

    def test_extrapolation_uses_observed_data_ranges(self):
        core = self.core()
        profile = dict(core.DEFAULT_PROFILE, Hours_Studied=50)
        data = pd.DataFrame({key: [value, value] for key, value in core.DEFAULT_PROFILE.items()})
        self.assertEqual(core.outside_observed_ranges(profile, data), ["Study hours"])


if __name__ == "__main__":
    unittest.main()
