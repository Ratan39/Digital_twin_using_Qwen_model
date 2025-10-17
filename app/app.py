
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Recommendations Dashboard", layout="wide")
st.title("🎓 Course Recommendations Dashboard")

# ---------------------------
# Data Loading
# ---------------------------
@st.cache_data
def load_data(default_path="/outputs/recommendation.csv"):
    # Try default path; fall back to empty df if not found.
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        return df
    else:
        return pd.DataFrame()

# ---------------------------
# Sidebar: Uploader + Filters
# ---------------------------
st.sidebar.subheader("Data")
uploaded = st.sidebar.file_uploader(
    "Upload recommendations.csv",
    type=["csv"],
    help="Drag & drop your CSV here or click to browse."
)

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # fallback to default path if present
    df = load_data()

st.sidebar.markdown("---")
st.sidebar.header("Filters")

if df.empty:
    st.info("No data found. Please upload your **recommendations.csv** in the sidebar to continue.")
    st.stop()

# Ensure expected columns exist
expected_cols = {
    "student_id",
    "student_background",
    "student_academic_interests",
    "student_professional_interests",
    "student_previous_work_experience",
    "course",
    "rank",
    "similarity",
    "confidence_percent",
    "answer_yes_no",
    "rationale",
}
missing = expected_cols - set(map(str, df.columns))
if missing:
    st.error(f"Missing expected columns: {sorted(missing)}")
    st.stop()

# Clean / coerce types
df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
df["confidence_percent"] = pd.to_numeric(df["confidence_percent"], errors="coerce")
df["answer_yes_no"] = df["answer_yes_no"].astype(str)

students = sorted(df["student_id"].unique().tolist())
courses = sorted(df["course"].unique().tolist())
answers = sorted(df["answer_yes_no"].unique().tolist())

student_sel = st.sidebar.multiselect("Students", students, default=students)
course_sel  = st.sidebar.multiselect("Courses", courses, default=courses)
answer_sel  = st.sidebar.multiselect("Answer (Yes/No)", answers, default=answers)

min_sim, max_sim = float(np.nanmin(df["similarity"])), float(np.nanmax(df["similarity"]))
min_conf, max_conf = float(np.nanmin(df["confidence_percent"])), float(np.nanmax(df["confidence_percent"]))

sim_range = st.sidebar.slider("Similarity range", min_value=0.0, max_value=1.0, value=(max(0.0, round(min_sim, 2)), min(1.0, round(max_sim, 2))), step=0.01)
conf_range = st.sidebar.slider("Confidence % range", min_value=0.0, max_value=100.0, value=(max(0.0, round(min_conf, 1)), min(100.0, round(max_conf, 1))), step=1.0)

topk = st.sidebar.number_input("Limit to top-K rank per student (0 = no limit)", min_value=0, max_value=50, value=0, step=1)

# Apply filters
fdf = df[
    df["student_id"].isin(student_sel)
    & df["course"].isin(course_sel)
    & df["answer_yes_no"].isin(answer_sel)
    & (df["similarity"].between(sim_range[0], sim_range[1]))
    & (df["confidence_percent"].between(conf_range[0], conf_range[1]))
].copy()

if topk and topk > 0:
    fdf = fdf.sort_values(["student_id", "rank"], ascending=[True, True]).groupby("student_id").head(int(topk))

# ---------------------------
# KPI Cards (Static + Filter-aware)
# ---------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Students", fdf["student_id"].nunique())
with col2:
    st.metric("Total Recs", len(fdf))
with col3:
    yes_rate = (fdf["answer_yes_no"].str.lower() == "yes").mean() if len(fdf) else 0.0
    st.metric("% Yes", f"{yes_rate*100:.1f}%")
with col4:
    st.metric("Avg Similarity", f"{fdf['similarity'].mean():.2f}")
with col5:
    st.metric("Avg Confidence", f"{fdf['confidence_percent'].mean():.1f}%")

st.markdown("---")

# ================================
# TABS: Non-Technical, Technical
# ================================
tab_nontech, tab_tech = st.tabs(["🧭 Non-Technical Overview", "🧪 Technical Overview"])

# ---------------------------
# Non-Technical Overview
# ---------------------------
with tab_nontech:
    # 1) Yes/No pie chart
    yn_counts = (
        fdf.groupby("answer_yes_no", dropna=False)
           .size()
           .reset_index(name="count")
    )
    pie_yesno = alt.Chart(yn_counts).mark_arc().encode(
        theta="count:Q",
        color=alt.Color("answer_yes_no:N", legend=alt.Legend(title="Answer")),
        tooltip=["answer_yes_no", "count"]
    ).properties(title="Yes vs No (count)")
    st.altair_chart(pie_yesno, use_container_width=True)

    # 2) Average similarity by course (bar)
    agg_course = (
        fdf.groupby("course", as_index=False)
           .agg(avg_similarity=("similarity","mean"),
                avg_confidence=("confidence_percent","mean"),
                count=("course","count"))
    )
    bar_avg_sim = alt.Chart(agg_course).mark_bar().encode(
        x=alt.X("course:N", sort="-y", title="Course"),
        y=alt.Y("avg_similarity:Q", title="Avg Similarity"),
        tooltip=["course", alt.Tooltip("avg_similarity:Q", format=".2f")]
    ).properties(title="Average Similarity by Course")
    st.altair_chart(bar_avg_sim, use_container_width=True)

    # 3) Number of recommendations (bar)
    counts = fdf["course"].value_counts().reset_index()
    counts.columns = ["course","count"]
    bar_counts = alt.Chart(counts).mark_bar().encode(
        x=alt.X("course:N", sort="-y", title="Course"),
        y=alt.Y("count:Q", title="Number of Recommendations"),
        tooltip=["course","count"]
    ).properties(title="Number of Recommendations by Course")
    st.altair_chart(bar_counts, use_container_width=True)

    # 4) % Yes for course (bar)
    yes_by_course = (
        fdf.assign(is_yes=(fdf["answer_yes_no"].str.lower()=="yes").astype(int))
           .groupby("course", as_index=False)["is_yes"].mean()
    )
    yes_by_course["yes_pct"] = yes_by_course["is_yes"] * 100
    bar_yes_pct = alt.Chart(yes_by_course).mark_bar().encode(
        x=alt.X("course:N", sort="-y", title="Course"),
        y=alt.Y("yes_pct:Q", title="% Yes"),
        tooltip=["course", alt.Tooltip("yes_pct:Q", format=".1f")]
    ).properties(title="% Yes by Course")
    st.altair_chart(bar_yes_pct, use_container_width=True)

    # 5) Confidence percent for courses (mean confidence by course)
    bar_conf = alt.Chart(agg_course).mark_bar().encode(
        x=alt.X("course:N", sort="-y", title="Course"),
        y=alt.Y("avg_confidence:Q", title="Avg Confidence %"),
        tooltip=["course", alt.Tooltip("avg_confidence:Q", format=".1f")]
    ).properties(title="Average Confidence % by Course")
    st.altair_chart(bar_conf, use_container_width=True)

# ---------------------------
# Technical Overview
# ---------------------------
with tab_tech:
    # 1) Search rationales with filter
    st.subheader("Search Rationales")
    q = st.text_input("Filter rationales by keyword (case-insensitive):", key="rationale_query")
    rdf = fdf.copy()
    if q.strip():
        rdf = rdf[rdf["rationale"].str.contains(q, case=False, na=False)]
    rdf_sorted = rdf.sort_values(["student_id", "rank"]).reset_index(drop=True)
    st.dataframe(
        rdf_sorted[["student_id","course","rank","answer_yes_no","similarity","confidence_percent","rationale"]],
        use_container_width=True
    )

    # 2) Keyword frequency
    st.subheader("Keyword Frequency")
    def tokenize(text):
        tokens = re.findall(r"[A-Za-z]{3,}", str(text).lower())
        stop = {
            "the","and","for","with","this","that","from","into","about","your","you","our","their","they","them",
            "were","was","are","is","his","her","she","him","out","all","any","not","have","has","had","get","got",
            "over","more","most","less","very","just","make","made","other","each","per","much","many","also","while",
            "where","when","what","which","who","whom","whose","shall","should","would","could","might","will","can",
            "because","since","toward","towards","using","use","used","under","between","among","across","onto","within",
            "without","above","below"
        }
        return [t for t in tokens if t not in stop]

    tokens = []
    for txt in rdf_sorted["rationale"].dropna().tolist():
        tokens.extend(tokenize(txt))
    freq = pd.DataFrame(Counter(tokens).most_common(40), columns=["term","count"]) if tokens else pd.DataFrame(columns=["term","count"])
    if not freq.empty:
        chart_kw = alt.Chart(freq).mark_bar().encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("term:N", sort="-x", title="Keyword"),
            tooltip=["term","count"]
        ).properties(title="Top Keywords in Rationales")
        st.altair_chart(chart_kw, use_container_width=True)
    else:
        st.info("No keywords to display for current filters.")

    # 3) Heatmap student x courses (Avg Similarity)
    st.subheader("Student × Course Heatmap (Avg Similarity)")
    heat = (
        fdf.groupby(["student_id","course"], as_index=False)
           .agg(avg_similarity=("similarity","mean"))
    )
    chart_heat = alt.Chart(heat).mark_rect().encode(
        x=alt.X("course:N", title="Course"),
        y=alt.Y("student_id:N", title="Student"),
        tooltip=["student_id","course", alt.Tooltip("avg_similarity:Q", format=".2f")],
        color=alt.Color("avg_similarity:Q", scale=alt.Scale(scheme="blues"), title="Avg Similarity")
    ).properties(height=400)
    st.altair_chart(chart_heat, use_container_width=True)

    # 4) N-grams for rationales (bigrams/trigrams)
    st.subheader("N-grams in Rationales")
    ngram_n = st.radio("Choose n-gram size", [2,3], index=0, horizontal=True, key="ngram_size")

    def ngrams_from_tokens(tokens_list, n=2):
        for i in range(len(tokens_list) - n + 1):
            yield " ".join(tokens_list[i:i+n])

    from collections import Counter as Ctr
    ngram_counts = Ctr()
    for txt in rdf_sorted["rationale"].dropna().tolist():
        toks = tokenize(txt)
        ngram_counts.update(ngrams_from_tokens(toks, n=int(ngram_n)))

    ngram_df = pd.DataFrame(ngram_counts.most_common(30), columns=["ngram","count"]) if ngram_counts else pd.DataFrame(columns=["ngram","count"])
    if not ngram_df.empty:
        chart_ng = alt.Chart(ngram_df).mark_bar().encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("ngram:N", sort="-x", title="N-gram"),
            tooltip=["ngram","count"]
        ).properties(title=f"Top {int(ngram_n)}-grams in Rationales")
        st.altair_chart(chart_ng, use_container_width=True)
    else:
        st.info("No n-grams to display for current filters.")

    # 5) Similarity box plots by course
    st.subheader("Similarity Distribution by Course (Box Plots)")
    box = alt.Chart(fdf).mark_boxplot().encode(
        x=alt.X("course:N", sort="-y", title="Course"),
        y=alt.Y("similarity:Q", title="Similarity"),
        tooltip=["course"]
    ).properties(title="Similarity Distribution by Course")
    st.altair_chart(box, use_container_width=True)
