import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Rationale Analytics From Qwen Model", layout="wide")

DEFAULT_CSV_PATHS = [
    "outputs/recommendations_with_professional_experience.csv",
    "outputs/recommendations.csv",
    "./recommendations_with_professional_experience.csv",
    "./recommendations.csv",
    "/mnt/data/recommendations_with_professional_experience.csv",
    "/mnt/data/recommendations.csv",
]

# Optional built-in student gender mapping (used if 'gender' column missing)
FALLBACK_STUDENT_GENDER = {
    "student_1": "Female",
    "student_2": "Male",
    "student_3": "Female",
    "student_4": "Male",
    "student_5": "Female",
    "student_6": "Male",
    "student_7": "Female",
    "student_8": "Male",
    "student_9": "Female",
    "student_10": "Male",
}

# ---------------------
# Helpers
# ---------------------
def load_recs(csv_path: str | None = None) -> pd.DataFrame:
    paths = [csv_path] if csv_path else DEFAULT_CSV_PATHS
    for p in paths:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            # normalize columns
            for c in ["answer_yes_no","student_id","course","rationale","student_background",
                      "student_interests","previous_work_experience","gender"]:
                if c in df.columns:
                    df[c] = df[c].astype(str)
            if "answer_yes_no" in df.columns:
                df["answer_yes_no"] = df["answer_yes_no"].str.strip().str.title()
            # ensure similarity numeric
            if "similarity" in df.columns:
                df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
            return df
    raise FileNotFoundError("Could not find a recommendations CSV.\n"
                            "Tried: " + ", ".join(paths))

def clean_na(series):
    return series.fillna("").astype(str)

def tokenize_count(texts, ngram_range=(1,1), max_features=None):
    # Use built-in 'english' stop words for compatibility across sklearn versions
    vect = CountVectorizer(lowercase=True, ngram_range=ngram_range,
                           stop_words='english', max_features=max_features)
    X = vect.fit_transform(texts)
    vocab = np.array(vect.get_feature_names_out())
    counts = np.asarray(X.sum(axis=0)).ravel()
    df = pd.DataFrame({"token": vocab, "count": counts}).sort_values("count", ascending=False)
    return df

def top_terms_series(texts, topn=30, n=1):
    if len(texts) == 0:
        return pd.DataFrame(columns=["token","count"])
    df = tokenize_count(texts, ngram_range=(n,n))
    return df.head(topn)

def yes_no_token_overlap(yes_texts, no_texts, topn=50):
    yes_df = tokenize_count(yes_texts, ngram_range=(1,1))
    no_df  = tokenize_count(no_texts,  ngram_range=(1,1))
    yes_top = set(yes_df.head(topn)["token"])
    no_top  = set(no_df.head(topn)["token"])
    overlap = sorted(list(yes_top & no_top))
    return overlap, yes_df.head(topn), no_df.head(topn)

def agg_interest_vs_course(df, interest_col: str):
    """Aggregate counts/yes/no/yes_rate by interest_col x course."""
    if interest_col not in df.columns or "course" not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp[interest_col] = tmp[interest_col].fillna("Unknown").astype(str)
    tmp["course"] = tmp["course"].fillna("Unknown").astype(str)
    tmp["is_yes"] = (tmp["answer_yes_no"].str.lower() == "yes")
    g = tmp.groupby([interest_col, "course"]).agg(
        count=("course","size"),
        yes=("is_yes","sum")
    ).reset_index()
    g["no"] = g["count"] - g["yes"]
    g["yes_rate"] = np.where(g["count"]>0, g["yes"]/g["count"], 0.0)
    return g

def heatmap_interest_course(agg_df, interest_col: str, metric: str, title: str):
    if agg_df.empty:
        return None
    # Pivot for heatmap
    pivot = agg_df.pivot(index=interest_col, columns="course", values=metric).fillna(0)
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        title=title
    )
    fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    return fig

# ---------------------
# Sidebar
# ---------------------
st.sidebar.title("⚙️ Controls")

csv_input = st.sidebar.text_input("CSV path (optional)")
df = load_recs(csv_input)

# If gender is missing, enrich from fallback mapping
if "gender" not in df.columns:
    df["gender"] = df.get("student_id", pd.Series(index=df.index, dtype=str)).map(FALLBACK_STUDENT_GENDER)

# Filters
students = sorted(df["student_id"].unique().tolist()) if "student_id" in df.columns else []
courses  = sorted(df["course"].unique().tolist()) if "course" in df.columns else []
answers  = sorted(df["answer_yes_no"].dropna().str.title().unique().tolist()) if "answer_yes_no" in df.columns else []

student_sel = st.sidebar.multiselect("Students", students, default=students if students else [])
course_sel  = st.sidebar.multiselect("Courses", courses, default=courses if courses else [])
answer_sel  = st.sidebar.multiselect("Answer (Yes/No)", answers if answers else ["Yes","No"],
                                     default=answers if answers else ["Yes","No"])

min_sim = 0.0
if "similarity" in df.columns:
    min_sim = st.sidebar.slider("Min similarity filter", 0.0, 1.0, 0.0, 0.01)
else:
    st.sidebar.caption("No 'similarity' column detected; skipping similarity filter.")

# Prepare filtered frame
flt = df.copy()
if student_sel:
    flt = flt[flt["student_id"].isin(student_sel)]
if course_sel:
    flt = flt[flt["course"].isin(course_sel)]
if answer_sel and "answer_yes_no" in flt.columns:
    flt = flt[flt["answer_yes_no"].isin([a.title() for a in answer_sel])]
if "similarity" in flt.columns:
    flt = flt[flt["similarity"] >= min_sim]

# Convenience subsets
yes_df = flt[flt["answer_yes_no"]=="Yes"] if "answer_yes_no" in flt.columns else flt.iloc[0:0]
no_df  = flt[flt["answer_yes_no"]=="No"]  if "answer_yes_no" in flt.columns else flt.iloc[0:0]

# ---------------------
# Header & KPIs
# ---------------------
st.title("🧠 Rationale Analytics For Response Generated Through Qwen Model")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(flt))
col2.metric("Students", flt["student_id"].nunique() if "student_id" in flt.columns else 0)
col3.metric("Courses", flt["course"].nunique() if "course" in flt.columns else 0)
if "answer_yes_no" in flt.columns and len(flt)>0:
    yes_rate = (flt["answer_yes_no"].str.lower() == "yes").mean()
    col4.metric("Yes Rate", f"{yes_rate*100:.1f}%")
else:
    col4.metric("Yes Rate", "—")

st.markdown("---")

# ---------------------
# Section A: Word Frequency with 1–5 grams
# ---------------------
st.header("1) Word Frequency (Unigram → 5-gram)")
texts_all = clean_na(flt["rationale"]) if "rationale" in flt.columns else pd.Series([], dtype=str)

top_uni = top_terms_series(texts_all, topn=30, n=1)
top_bi  = top_terms_series(texts_all, topn=20, n=2)
top_tri = top_terms_series(texts_all, topn=15, n=3)
top_4g  = top_terms_series(texts_all, topn=10, n=4)
top_5g  = top_terms_series(texts_all, topn=8,  n=5)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Top Unigrams")
    if not top_uni.empty:
        fig = px.bar(top_uni, x="token", y="count", title="Top 30 words")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rationale text found.")
with c2:
    st.subheader("Top Bigrams")
    if not top_bi.empty:
        fig = px.bar(top_bi, x="token", y="count", title="Top 20 phrases")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rationale text found.")

c3, c4, c5 = st.columns(3)
with c3:
    st.subheader("Top Trigrams")
    if not top_tri.empty:
        fig = px.bar(top_tri, x="token", y="count", title="Top 15 trigrams")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trigrams found.")
with c4:
    st.subheader("Top 4-grams")
    if not top_4g.empty:
        fig = px.bar(top_4g, x="token", y="count", title="Top 10 four-grams")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 4-grams found.")
with c5:
    st.subheader("Top 5-grams")
    if not top_5g.empty:
        fig = px.bar(top_5g, x="token", y="count", title="Top 8 five-grams")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 5-grams found.")

st.markdown("---")

# ---------------------
# Section B: Keyword Overlap (Yes vs No) — kept as-is
# ---------------------
st.header("2) Keyword Overlap (Yes vs No)")
if "answer_yes_no" in flt.columns and not flt.empty:
    yes_texts = clean_na(yes_df["rationale"])
    no_texts  = clean_na(no_df["rationale"])
    overlap, yes_top, no_top = yes_no_token_overlap(yes_texts, no_texts, topn=50)
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        st.subheader("Top Yes Tokens")
        st.dataframe(yes_top.reset_index(drop=True))
    with colB:
        st.subheader("Top No Tokens")
        st.dataframe(no_top.reset_index(drop=True))
    with colC:
        st.subheader("Overlap (Top 50 ∩ Top 50)")
        st.write(overlap if overlap else "No overlap within top-50 sets.")
else:
    st.info("Yes/No labels not available.")

st.markdown("---")

# ---------------------
# Section C: Similarity vs Yes/No
# ---------------------
st.header("3) Similarity vs Yes/No")
if "similarity" in flt.columns and "answer_yes_no" in flt.columns and not flt.empty:
    fig = px.box(flt, x="answer_yes_no", y="similarity", color="answer_yes_no",
                 title="Similarity distribution by decision (Yes/No)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------
# Section D: Course vs Similarity
# ---------------------
st.header("4) Course vs Similarity")
if "course" in flt.columns and "similarity" in flt.columns and not flt.empty:
    # Boxplot per course (optional for distribution)
    st.subheader("Distribution by Course")
    figb = px.box(flt, x="course", y="similarity", points="outliers",
                  title="Similarity distribution per course")
    st.plotly_chart(figb, use_container_width=True)
else:
    st.info("Need 'course' and 'similarity' columns to show this section.")

st.markdown("---")

# ---------------------
# Section G: Academic Interests vs Course
# ---------------------
st.header("7) Academic Interests vs Course")
ACADEMIC_COL = "student_academic_interests" if "student_academic_interests" in flt.columns else ("student_interests" if "student_interests" in flt.columns else None)
if ACADEMIC_COL and ACADEMIC_COL in flt.columns and "course" in flt.columns and not flt.empty:
    agg_acad = agg_interest_vs_course(flt, ACADEMIC_COL)

    metric2 = st.selectbox("Heatmap metric (Academic Interests)", ["yes", "no", "count", "yes_rate"], index=0)
    fig_acad = heatmap_interest_course(agg_acad, ACADEMIC_COL, metric2, f"{ACADEMIC_COL} × Course — {metric2}")
    if fig_acad:
        st.plotly_chart(fig_acad, use_container_width=True)


# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.caption("Use the sidebar filters to subset students/courses/decisions and analyze rationale patterns and similarity behavior.")
