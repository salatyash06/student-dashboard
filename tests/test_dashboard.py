"""Integration checks for the actual Streamlit UI and original saved model."""
import unittest
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("navigation", [widget.key for widget in app.radio],
                      "Dashboard navigation has not been implemented")
        return app

    def test_prediction_preserves_original_model_and_records_submitted_profile(self):
        app = self.app()
        app.slider(key="main_Hours_Studied").set_value(25)
        app.button(key="FormSubmitter:profile-Predict score").click().run()
        self.assertFalse(app.exception)
        history = app.session_state["history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["Hours_Studied"], 25)
        features = joblib.load(ROOT / "models/features.pkl")
        model = joblib.load(ROOT / "models/model.pkl")
        original = pd.DataFrame({"Hours_Studied": [25], "Attendance": [80],
            "Previous_Scores": [75], "Sleep_Hours": [7], "Tutoring_Sessions": [1]})
        expected = float(model.predict(original.reindex(columns=features, fill_value=0))[0])
        self.assertAlmostEqual(history[0]["Predicted_Score"], expected)
        app.radio(key="navigation").set_value("Prediction history").run()
        self.assertEqual(len(app.dataframe[0].value), 1)
        app.button(key="clear_history").click().run()
        self.assertEqual(app.session_state["history"], [])

    def test_reset_restores_profile_and_does_not_add_history(self):
        app = self.app()
        app.slider(key="main_Hours_Studied").set_value(35)
        app.button(key="FormSubmitter:profile-Predict score").click().run()
        app.button(key="reset_profile").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.slider(key="main_Hours_Studied").value, 20)
        self.assertEqual(len(app.session_state["history"]), 1)

    def test_compare_identical_profiles_has_zero_difference(self):
        app = self.app()
        app.radio(key="navigation").set_value("What-if lab").run()
        app.slider(key="compare_b_Hours_Studied").set_value(20)
        app.slider(key="compare_b_Attendance").set_value(80)
        app.button(key="compare_profiles").click().run()
        self.assertFalse(app.exception)
        result = app.session_state["comparison"]
        self.assertAlmostEqual(result["scores"][0], result["scores"][1])
        self.assertEqual(len(app.session_state["history"]), 0)

    def test_explorer_handles_empty_filters_and_restores_data(self):
        app = self.app()
        app.radio(key="navigation").set_value("Data explorer").run()
        self.assertFalse(app.exception)
        app.multiselect(key="school_filter").set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("No students" in x.value for x in app.info))
        app.button(key="reset_filters").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 6607)

    def test_empty_history_returns_to_prediction_studio(self):
        app = self.app()
        app.radio(key="navigation").set_value("Prediction history").run()
        next(b for b in app.button if b.label == "Create a prediction").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="navigation").value, "Prediction studio")

    def test_comparison_marks_results_stale_after_edit(self):
        app = self.app()
        app.radio(key="navigation").set_value("What-if lab").run()
        app.button(key="compare_profiles").click().run()
        self.assertGreater(app.session_state["comparison"]["scores"][1],
                           app.session_state["comparison"]["scores"][0])
        app.slider(key="compare_b_Hours_Studied").set_value(30).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Inputs have changed" in info.value for info in app.info))


if __name__ == "__main__":
    unittest.main()
