"""Scholar: a light, interactive frontend for the original student-score model."""
from datetime import datetime, timezone
from pathlib import Path
import math

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_core import DEFAULT_PROFILE, FIELDS, outside_observed_ranges, prepare_input

ROOT = Path(__file__).resolve().parent
PURPLE = "#6558df"
TEAL = "#23a894"
CHART_COLORS = [PURPLE, TEAL, "#e8a24b"]
st.set_page_config(page_title="Scholar | Student Performance", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(ROOT / 'assets/style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models/model.pkl"), joblib.load(ROOT / "models/features.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data/StudentPerformanceFactors.csv")


try:
    model, features = load_model()
    data = load_data()
except Exception:
    st.error("The project data or model could not be loaded. Keep the models and data folders "
             "beside app.py and install the versions in requirements.txt.")
    st.stop()

st.session_state.setdefault("history", [])
st.session_state.setdefault("active_profile", dict(DEFAULT_PROFILE))
st.session_state.setdefault("result", None)
st.session_state.setdefault("comparison", None)


def predict(profile):
    score = float(model.predict(prepare_input(profile, features))[0])
    if not math.isfinite(score):
        raise ValueError("The model returned a non-finite score.")
    return score


def reset_profile():
    for field, value in DEFAULT_PROFILE.items():
        st.session_state[f"main_{field}"] = value
    st.session_state.active_profile = dict(DEFAULT_PROFILE)
    st.session_state.result = None


def reset_filters():
    st.session_state.school_filter = sorted(data.School_Type.dropna().unique().tolist())
    st.session_state.motivation_filter = sorted(data.Motivation_Level.dropna().unique().tolist())
    st.session_state.score_filter = (int(data.Exam_Score.min()), int(data.Exam_Score.max()))


def open_studio():
    st.session_state.navigation = "Prediction studio"


def panel(title, caption, symbol="◈"):
    st.markdown(f'<div class="panel-heading"><span class="panel-icon">{symbol}</span>'
                f'<h3>{title}</h3></div><div class="panel-caption">{caption}</div>',
                unsafe_allow_html=True)


def heading(title, subtitle):
    st.markdown(f'<h1 class="page-title">{title}</h1><div class="page-subtitle">{subtitle}</div>',
                unsafe_allow_html=True)


def chart(figure, key, height=280):
    figure.update_layout(template="plotly_white", height=height,
        margin=dict(l=10, r=15, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color="#6b7690", size=11),
        colorway=CHART_COLORS, legend=dict(orientation="h", y=1.12, x=0),
        hoverlabel=dict(bgcolor="white", font_color="#26324b"))
    figure.update_xaxes(gridcolor="#edf0f6", zeroline=False)
    figure.update_yaxes(gridcolor="#edf0f6", zeroline=False)
    st.plotly_chart(figure, key=key, theme=None, width="stretch",
                    config={"displaylogo": False, "scrollZoom": False,
                            "toImageButtonOptions": {"filename": key, "scale": 2}})


def profile_controls(prefix, defaults):
    values = {}
    for field, (label, low, high, help_text) in FIELDS.items():
        label = {"Hours_Studied": "Study hours / week", "Attendance": "Attendance (%)",
                 "Sleep_Hours": "Sleep hours / night"}.get(field, label)
        key = f"{prefix}_{field}"
        if key not in st.session_state:
            st.session_state[key] = defaults[field]
        values[field] = st.slider(label, low, high, key=key, help=help_text)
    return values


def show_extrapolation(profile):
    outside = outside_observed_ranges(profile, data)
    if outside:
        st.warning("Outside the dataset's observed range: " + ", ".join(outside)
                   + ". This prediction extrapolates beyond the observed data.")


def assumptions():
    with st.expander("How to interpret these predictions"):
        st.write("The original model uses 27 encoded features. This form supplies the same five "
                 "numeric inputs as your original app; the remaining features are fixed at zero. "
                 "Predictions are estimates, and profile comparisons are not evidence of cause and effect.")
        st.caption("No confidence interval or validated accuracy is available in the supplied app. "
                   "The original model and its preprocessing are preserved.")


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">S</div><div>'
                '<div class="brand-name">scholar<span style="color:#6558df">.</span></div>'
                '<div class="brand-sub">PERFORMANCE STUDIO</div></div></div>'
                '<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
    page = st.radio("Workspace navigation", ["Prediction studio", "What-if lab", "Data explorer", "Prediction history"],
                    label_visibility="collapsed", key="navigation")
    st.markdown('<div class="sidebar-note"><strong>Small changes. Clearer insights.</strong>'
                'Explore learning patterns, compare possibilities, and bring your data science project to life.'
                '</div>', unsafe_allow_html=True)
    st.caption("PYTHON FOR DATA SCIENCE")
    st.caption("Student performance project · Light edition")

st.markdown(f'<div class="topline"><span>Workspace &nbsp;/&nbsp; <strong>{page}</strong></span>'
            '<span class="status">● &nbsp; Local dataset connected</span></div>', unsafe_allow_html=True)


if page == "Prediction studio":
    st.markdown('<div class="hero"><div class="eyebrow">A clearer view of student performance</div>'
                '<h1>Turn learning habits into insights.</h1>'
                '<p>Explore how study patterns relate to exam performance. Build a student profile, '
                'estimate a score, and discover what could change.</p></div>', unsafe_allow_html=True)
    columns = st.columns(4)
    columns[0].metric("Students in dataset", f"{len(data):,}")
    columns[1].metric("Average exam score", f"{data.Exam_Score.mean():.1f}")
    columns[2].metric("Average attendance", f"{data.Attendance.mean():.0f}%")
    columns[3].metric("Typical weekly study", f"{data.Hours_Studied.median():.0f} hrs")
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.form("profile"):
            panel("Build a student profile", "Adjust the five inputs, then generate your prediction.", "◇")
            profile = profile_controls("main", st.session_state.active_profile)
            submitted = st.form_submit_button("Predict score", type="primary", width="stretch")
        st.button("Reset to example profile", key="reset_profile", on_click=reset_profile,
                  icon=":material/restart_alt:")
        if submitted:
            try:
                score = predict(profile)
                st.session_state.active_profile = dict(profile)
                st.session_state.result = {"profile": dict(profile), "score": score}
                st.session_state.history.append({"Created_UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                **profile, "Predicted_Score": score})
                st.session_state.history = st.session_state.history[-100:]
            except (ValueError, TypeError) as error:
                st.error(f"Could not calculate a prediction: {error}")
        assumptions()
    with right:
        result = st.session_state.result
        result_profile = result["profile"] if result else DEFAULT_PROFILE
        score = result["score"] if result else predict(DEFAULT_PROFILE)
        difference = score - data.Exam_Score.mean()
        badge = "SAVED TO SESSION" if result else "EXAMPLE PREVIEW"
        st.markdown(f'<div class="score-card"><div class="score-top"><div class="eyebrow" style="margin:0">'
                    f'Predicted exam score</div><span class="pill">{badge}</span></div>'
                    f'<div class="score-number">{score:.1f}<span> / 100</span></div>'
                    f'<div class="score-description">{abs(difference):.1f} points '
                    f'{"above" if difference >= 0 else "below"} the dataset average</div>'
                    f'<div class="score-track"><div class="score-fill" style="width:{min(100,max(0,score)):.1f}%"></div></div>'
                    '<div class="score-description">An estimate from your project’s trained model.</div></div>',
                    unsafe_allow_html=True)
        st.caption(f"{'Last submitted profile' if result else 'Example profile'} · "
                   f"{result_profile['Hours_Studied']} study hrs · {result_profile['Attendance']}% attendance · "
                   f"{result_profile['Previous_Scores']} previous score · {result_profile['Sleep_Hours']} sleep hrs · "
                   f"{result_profile['Tutoring_Sessions']} tutoring sessions")
        if result:
            st.caption("Press Predict score to apply changes in the form and save another result.")
        show_extrapolation(result_profile)
        if not 0 <= score <= 100:
            st.warning("This model estimate falls outside 0–100. The displayed score is not clipped.")
        with st.container(border=True):
            panel("Where does this score sit?", "Actual exam scores across the full student dataset.", "↗")
            fig = px.histogram(data, x="Exam_Score", nbins=32, color_discrete_sequence=["#b1aaf3"])
            fig.add_vline(x=score, line_color=PURPLE, line_width=2, line_dash="dash")
            fig.update_layout(bargap=.13, showlegend=False, xaxis_title="Exam score", yaxis_title="Students")
            chart(fig, "score-distribution", 235)
            st.caption(f"Dashed line: model estimate ({score:.1f}). Dataset mean: {data.Exam_Score.mean():.1f}.")
    st.markdown('<div class="insight"><strong>Explore the possibilities →</strong> '
                'Open What-if lab to compare two profiles, or Data explorer to investigate the patterns behind the numbers.</div>',
                unsafe_allow_html=True)

elif page == "What-if lab":
    heading("One profile. More possibilities.", "Compare two scenarios and see how the same model responds to different inputs.")
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        panel("Profile A · Baseline", "Your starting point for comparison.", "A")
        a = profile_controls("compare_a", DEFAULT_PROFILE)
    with right, st.container(border=True):
        panel("Profile B · What if?", "Try a different study pattern.", "B")
        b = profile_controls("compare_b", dict(DEFAULT_PROFILE, Hours_Studied=25, Attendance=90))
    if st.button("Compare profiles", key="compare_profiles", type="primary", icon=":material/compare_arrows:"):
        try:
            st.session_state.comparison = {"profiles": [dict(a), dict(b)], "scores": [predict(a), predict(b)]}
        except (ValueError, TypeError) as error:
            st.error(f"Could not compare profiles: {error}")
    comparison = st.session_state.comparison
    if comparison:
        if comparison["profiles"] != [a, b]:
            st.info("Inputs have changed. Click Compare profiles to update the results below.")
        sa, sb = comparison["scores"]
        cols = st.columns(3)
        cols[0].metric("Profile A estimate", f"{sa:.2f}")
        cols[1].metric("Profile B estimate", f"{sb:.2f}")
        cols[2].metric("Estimated change · B − A", f"{sb-sa:+.2f} pts")
        left, right = st.columns([1, 1.1], gap="large")
        with left, st.container(border=True):
            panel("Side by side", "Model estimates for the last compared profiles.")
            fig = go.Figure(go.Bar(x=["Profile A", "Profile B"], y=[sa, sb],
                marker_color=[PURPLE, TEAL], text=[f"{sa:.2f}", f"{sb:.2f}"], textposition="outside"))
            fig.update_layout(yaxis_title="Predicted score", yaxis_range=[min(0,sa,sb)-5, max(100,sa,sb)+8])
            chart(fig, "profile-comparison")
        with right, st.container(border=True):
            panel("What changed?", "Input differences in the last comparison.", "⇄")
            pa, pb = comparison["profiles"]
            table = pd.DataFrame({"Input": [FIELDS[k][0] for k in FIELDS],
                                  "Profile A": list(pa.values()), "Profile B": list(pb.values()),
                                  "Change": [pb[k]-pa[k] for k in FIELDS]})
            st.dataframe(table, hide_index=True, width="stretch")
            export = pd.DataFrame([{ "Profile": name, **p, "Predicted_Score": s}
                                   for name, p, s in zip(["A", "B"], [pa, pb], [sa, sb])])
            st.download_button("Download comparison", export.to_csv(index=False).encode("utf-8"),
                               "profile-comparison.csv", "text/csv", icon=":material/download:", on_click="ignore")
        for name, profile in zip(["A", "B"], comparison["profiles"]):
            if outside_observed_ranges(profile, data):
                st.caption(f"Profile {name}")
                show_extrapolation(profile)
        st.caption("Estimated changes describe the model's response; they are not guaranteed improvements or causal effects.")
    else:
        st.info("Set up both profiles and click Compare profiles to see the estimated difference.")
    assumptions()

elif page == "Data explorer":
    heading("Let the data tell its story.", "Filter the original student dataset. Every chart and download reflects your selection.")
    schools = sorted(data.School_Type.dropna().unique().tolist())
    motivations = sorted(data.Motivation_Level.dropna().unique().tolist())
    score_range = (int(data.Exam_Score.min()), int(data.Exam_Score.max()))
    st.session_state.setdefault("school_filter", schools)
    st.session_state.setdefault("motivation_filter", motivations)
    st.session_state.setdefault("score_filter", score_range)
    with st.container(border=True):
        panel("Refine your view", "Combine filters to explore a student group.", "≡")
        filters = st.columns([1, 1, 1.4])
        with filters[0]:
            selected_schools = st.multiselect("School type", schools, key="school_filter")
        with filters[1]:
            selected_motivation = st.multiselect("Motivation level", motivations, key="motivation_filter")
        with filters[2]:
            selected_scores = st.slider("Exam score range", *score_range, key="score_filter")
        st.button("Reset filters", key="reset_filters", on_click=reset_filters, icon=":material/restart_alt:")
    filtered = data[data.School_Type.isin(selected_schools) & data.Motivation_Level.isin(selected_motivation)
                    & data.Exam_Score.between(*selected_scores)]
    if filtered.empty:
        st.info("No students match these filters. Select a school and motivation level, widen the score range, or reset filters.")
    else:
        stats = st.columns(4)
        stats[0].metric("Matching students", f"{len(filtered):,}")
        stats[1].metric("Average exam score", f"{filtered.Exam_Score.mean():.1f}")
        stats[2].metric("Average study hours", f"{filtered.Hours_Studied.mean():.1f}")
        stats[3].metric("Share of dataset", f"{len(filtered)/len(data):.1%}")
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        left, right = st.columns([1.4, 1], gap="large")
        with left, st.container(border=True):
            panel("Learning patterns", "Hover for details. Zoom or download using the chart toolbar.", "↗")
            axis = st.selectbox("Compare exam scores against", ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Tutoring_Sessions"],
                                format_func=lambda value: FIELDS[value][0], key="scatter_axis")
            fig = px.scatter(filtered, x=axis, y="Exam_Score", color="Motivation_Level", opacity=.5,
                             color_discrete_map={"Low": "#e8a24b", "Medium": PURPLE, "High": TEAL},
                             labels={axis: FIELDS[axis][0], "Exam_Score": "Exam score", "Motivation_Level": "Motivation"},
                             hover_data=["School_Type", "Attendance"])
            fig.update_traces(marker_size=6)
            chart(fig, "learning-patterns", 340)
        with right, st.container(border=True):
            panel("Score distribution", "Exam scores in the selected student group.", "▥")
            fig = px.histogram(filtered, x="Exam_Score", nbins=25, color_discrete_sequence=[PURPLE])
            fig.update_layout(xaxis_title="Exam score", yaxis_title="Students", bargap=.15)
            chart(fig, "filtered-distribution", 420)
        with st.container(border=True):
            panel("Explore the records", f"{len(filtered):,} of {len(data):,} rows · Click a column header to sort.", "≡")
            st.dataframe(filtered, hide_index=True, width="stretch", height=350)
            st.download_button("Download filtered data", filtered.to_csv(index=False).encode("utf-8"),
                               "student-data-filtered.csv", "text/csv", icon=":material/download:", on_click="ignore")
        st.caption("Original values are preserved, including scores above 100. Observed associations do not establish causation.")

else:
    heading("Your prediction notebook.", "Revisit the profiles you submitted and take your results with you.")
    history = st.session_state.history
    if not history:
        with st.container(border=True):
            panel("A fresh start", "Your session history is empty.", "◷")
            st.write("Generate a prediction in Prediction studio to save your first result here.")
            st.button("Create a prediction", type="primary", on_click=open_studio)
    else:
        table = pd.DataFrame(history)
        stats = st.columns(3)
        stats[0].metric("Saved predictions", len(table))
        stats[1].metric("Latest estimate", f"{table.Predicted_Score.iloc[-1]:.2f}")
        stats[2].metric("Average estimate", f"{table.Predicted_Score.mean():.2f}")
        with st.container(border=True):
            panel("Session activity", "Most recent first · Timestamps are in UTC · Click a column header to sort.", "◷")
            st.dataframe(table.iloc[::-1], hide_index=True, width="stretch",
                         column_config={"Predicted_Score": st.column_config.NumberColumn("Predicted score", format="%.2f")})
            buttons = st.columns([1, 1, 2])
            with buttons[0]:
                st.download_button("Export history", table.to_csv(index=False).encode("utf-8"),
                                   "prediction-history.csv", "text/csv", icon=":material/download:", on_click="ignore")
            with buttons[1]:
                if st.button("Clear history", key="clear_history", icon=":material/delete_outline:"):
                    st.session_state.history = []
                    st.rerun()
    st.caption("History is held in this browser session only (latest 100 predictions). Export it before closing the session.")

st.markdown('<div class="footer"><span>scholar. &nbsp; / &nbsp; Student Performance Studio</span>'
            '<span>Python for Data Science &nbsp; · &nbsp; Explore. Compare. Understand.</span></div>', unsafe_allow_html=True)
