"""Small, UI-independent helpers. Preserve the supplied model's original encoding."""
import math

import pandas as pd

DEFAULT_PROFILE = {"Hours_Studied": 20, "Attendance": 80, "Previous_Scores": 75,
                   "Sleep_Hours": 7, "Tutoring_Sessions": 1}
FIELDS = {
    "Hours_Studied": ("Study hours", 0, 80, "Hours studied per week. Observed dataset range: 1–44."),
    "Attendance": ("Attendance", 0, 100, "Percentage of classes attended. Observed dataset range: 60–100%."),
    "Previous_Scores": ("Previous exam score", 0, 100, "Score on the previous exam. Observed dataset range: 50–100."),
    "Sleep_Hours": ("Sleep hours", 0, 24, "Average sleep per night. Observed dataset range: 4–10 hours."),
    "Tutoring_Sessions": ("Tutoring sessions", 0, 10, "Number of tutoring sessions. Observed dataset range: 0–8."),
}


def prepare_input(profile, features):
    for field, (label, low, high, _) in FIELDS.items():
        value = profile.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{label} must be a finite number from {low} to {high}.")
        if field == "Tutoring_Sessions" and int(value) != value:
            raise ValueError("Tutoring sessions must be a whole number.")
    # All five inputs are numeric; the original get_dummies call leaves them unchanged.
    return pd.DataFrame([{key: profile[key] for key in FIELDS}]).reindex(columns=features, fill_value=0)


def outside_observed_ranges(profile, data):
    return [FIELDS[key][0] for key, value in profile.items()
            if key in FIELDS and not data[key].min() <= value <= data[key].max()]
