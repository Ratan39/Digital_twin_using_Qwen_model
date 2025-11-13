import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter
import re

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Graduate Program Interest Dashboard",
    layout="wide",
)

st.title("🎓 Graduate Program Interest Dashboard")

st.markdown(
    """
This dashboard summarizes how simulated graduate students respond to different graduate programs 
(Yes/No interest, confidence, and rationale). Use the filters on the left and the tabs below to explore.
"""
)

# ---------------------------------
# Helpers
# ---------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        default_path = Path("recommendation.csv")
        if not default_path.exists():
            st.error(
                "No file uploaded and `recommendation.csv` not found in the current directory."
            )
            return None
        df = pd.read_csv(default_path)

    # Basic cleanup / typing
    for col in ["rank", "similarity", "confidence_percent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize answer column
    if "answer_yes_no" in df.columns:
        df["answer_yes_no"] = df["answer_yes_no"].astype(str).str.strip().str.title()

    return df


def simplify_category(text: str, default_label="Unknown"):
    """Simple heuristic to turn free text into a short category label."""
    if pd.isna(text):
        return default_label
    text = str(text).strip()
    if not text:
        return default_label
    tokens = text.split()
    return " ".join(tokens[:3])


def preprocess_rationales(rationales: pd.Series):
    """Lowercase, remove punctuation, and split into tokens per rationale."""
    texts = rationales.fillna("").astype(str).str.lower()
    texts = texts.apply(lambda t: re.sub(r"[^a-z0-9\s]", " ", t))
    token_lists = [t.split() for t in texts]
    return token_lists


STOPWORDS = set(
    """
a an the and or but if while is are was were be been being i you he she they we it this that these those
of for to in on at from with by about as into through during before after above below up down out over
so than too very can could will would should may might do does did done have has had having not no yes
just more most some any each few such own same other another much many really
""".split()
)


def ngram_counts(token_lists, n=1, top_n=20):
    """Compute top n-gram frequencies from a list of token lists."""
    counter = Counter()
    for tokens in token_lists:
        # Filter stopwords for unigrams; for bigrams/trigrams we allow them but you could tweak
        if n == 1:
            tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            counter[ngram] += 1
    most_common = counter.most_common(top_n)
    return pd.DataFrame(
        {
            "ngram": [" ".join(k) for k, _ in most_common],
            "count": [v for _, v in most_common],
        }
    )


# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.header("Data & Filters")

uploaded_file = st.sidebar.file_uploader(
    "Upload recommendation CSV",
    type=["csv"],
    help="If not provided, the app will look for recommendation.csv in the current folder.",
)

df = load_data(uploaded_file)

if df is None:
    st.stop()

# Ensure essential columns exist
required_cols = [
    "student_id",
    "course",
    "answer_yes_no",
    "similarity",
    "confidence_percent",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

all_programs = sorted(df["course"].dropna().unique().tolist())
all_students = sorted(df["student_id"].dropna().unique().tolist())
answer_options = sorted(df["answer_yes_no"].dropna().unique().tolist())

selected_programs = st.sidebar.multiselect(
    "Filter by program(s)",
    options=all_programs,
    default=all_programs,
)

selected_answers = st.sidebar.multiselect(
    "Filter by answer",
    options=answer_options,
    default=answer_options,
)

min_sim, max_sim = float(df["similarity"].min()), float(df["similarity"].max())
similarity_range = st.sidebar.slider(
    "Similarity range filter",
    min_value=float(round(min_sim, 2)),
    max_value=float(round(max_sim, 2)),
    value=(float(round(min_sim, 2)), float(round(max_sim, 2))),
    step=0.01,
)

# Similarity threshold for “high similarity” (used in confusion matrix)
sim_threshold = st.sidebar.slider(
    "Similarity threshold for 'High similarity' (Diagnostics)",
    min_value=float(round(min_sim, 2)),
    max_value=float(round(max_sim, 2)),
    value=float(round((min_sim + max_sim) / 2, 2)),
    step=0.01,
)

# Confidence threshold for "High-confidence Yes" in funnel
conf_threshold = st.sidebar.slider(
    "Confidence threshold for High-confidence Yes (Funnel)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0,
)

# Apply main filters
mask = (
    df["course"].isin(selected_programs)
    & df["answer_yes_no"].isin(selected_answers)
    & df["similarity"].between(similarity_range[0], similarity_range[1])
)
filtered = df[mask].copy()

if filtered.empty:
    st.warning("No data after applying filters. Try relaxing the filters on the left.")
    st.stop()

yes_only = filtered[filtered["answer_yes_no"] == "Yes"].copy()

# ---------------------------------
# Tabs layout (3 tabs)
# ---------------------------------
tab_overview, tab_deep, tab_rational = st.tabs(
    ["Overview", "Deeper Insights", "Rational Analysis (Qwen Model)"]
)

# ---------------------------------
# 1) Overview Tab
# ---------------------------------
with tab_overview:
    st.subheader("📌 Overview")

    # KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)

    total_students = filtered["student_id"].nunique()
    total_programs = filtered["course"].nunique()
    yes_rate = (
        (filtered["answer_yes_no"] == "Yes").mean() * 100
        if not filtered.empty
        else 0
    )
    avg_conf_yes = (
        yes_only["confidence_percent"].mean()
        if not yes_only.empty
        else np.nan
    )
    top_program = (
        yes_only["course"].value_counts().idxmax()
        if not yes_only.empty
        else "N/A"
    )

    col1.metric("Students in view", total_students)
    col2.metric("Programs in view", total_programs)
    col3.metric("Yes response rate", f"{yes_rate:.1f}%")
    col4.metric(
        "Avg. confidence (Yes only)",
        f"{avg_conf_yes:.1f}%" if not np.isnan(avg_conf_yes) else "N/A",
    )
    col5.metric("Most interested program", top_program)

    st.markdown("---")

    # Overall Yes/No pie chart
    st.markdown("### Overall Yes/No split")
    yesno_counts = (
    filtered["answer_yes_no"]
    .value_counts()
    .reset_index(name="count")
    .rename(columns={"index": "answer_yes_no"})
    )

    pie_fig = px.pie(
        yesno_counts,
        names="answer_yes_no",
        values="count",
        hole=0.3,
        color="answer_yes_no",
        color_discrete_map={
            "Yes": "#1b9e77",   # green
            "No": "#d95f02"     # orange
        }
    )
    st.plotly_chart(pie_fig, use_container_width=True)


    st.markdown("---")

    st.markdown("### Number of recommendations per course")

    rec_counts = (
        filtered
        .groupby("course", as_index=False)
        .agg(count=("course", "size"))
    )

    rec_counts = rec_counts.loc[:, ~rec_counts.columns.duplicated()]

    rec_bar = (
        alt.Chart(rec_counts)
        .mark_bar(color="#4C72B0")  # deep blue
        .encode(
            x=alt.X("course:N", sort="-y", title="Program"),
            y=alt.Y("count:Q", title="Number of recommendations"),
            tooltip=["course", "count"],
        )
        .properties(height=350)
    )

    st.altair_chart(rec_bar, use_container_width=True)



    st.markdown("---")

    # % of Yes vs course
    st.markdown("### % of Yes responses per course")
    course_yes_stats = (
        filtered.groupby("course")["answer_yes_no"]
        .value_counts(normalize=True)
        .rename("prop")
        .reset_index()
    )
    course_yes_stats = course_yes_stats[course_yes_stats["answer_yes_no"] == "Yes"]
    if not course_yes_stats.empty:
        course_yes_stats["prop_percent"] = course_yes_stats["prop"] * 100
        yes_pct_chart = (
            alt.Chart(course_yes_stats)
            .mark_bar()
            .encode(
                x=alt.X("course:N", sort="-y", title="Program"),
                y=alt.Y("prop_percent:Q", title="% Yes responses"),
                tooltip=[
                    "course",
                    alt.Tooltip("prop_percent:Q", format=".1f", title="% Yes"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(yes_pct_chart, use_container_width=True)
    else:
        st.info("No Yes responses in the current filter to compute % Yes per course.")

    st.markdown("---")

    # Program popularity (Yes/No stacked bar)
    st.markdown("### Program Popularity (Yes/No Responses)")

    program_counts = (
        filtered.groupby(["course", "answer_yes_no"])
        .size()
        .reset_index(name="count")
    )

    popularity_bar = (
        alt.Chart(program_counts)
        .mark_bar()
        .encode(
            y=alt.Y("course:N", sort="-x", title="Program"),
            x=alt.X("count:Q", title="Responses"),
            color=alt.Color(
                "answer_yes_no:N",
                title="Answer",
                scale=alt.Scale(
                    domain=["Yes", "No"],
                    range=["#1b9e77", "#d95f02"],  # green and orange again
                )
            ),
            tooltip=["course", "answer_yes_no", "count"],
        )
        .properties(height=400)
    )

    st.altair_chart(popularity_bar, use_container_width=True)


    st.markdown(
        """
This shows, for each program, how many Yes vs No responses it received in the current view.
"""
    )

# ---------------------------------
# 2) Deeper Insights Tab
# ---------------------------------
with tab_deep:
    st.subheader("🧠 Deeper Insights")

    # Confusion matrix (interest vs similarity threshold)
    st.markdown("#### Confusion Matrix (Interest vs Similarity Threshold)")

    diag_df = df.copy()
    diag_df["high_sim"] = diag_df["similarity"] >= sim_threshold
    diag_df["interest_yes"] = diag_df["answer_yes_no"] == "Yes"

    diag_df["sim_label"] = diag_df["high_sim"].map(
        {True: "High similarity", False: "Low similarity"}
    )
    diag_df["interest_label"] = diag_df["interest_yes"].map(
        {True: "Yes", False: "No"}
    )

    conf_counts = (
        diag_df.groupby(["sim_label", "interest_label"])
        .size()
        .reset_index(name="count")
    )

    conf_pivot = conf_counts.pivot(
        index="sim_label", columns="interest_label", values="count"
    ).fillna(0)

    st.write("Confusion matrix table:")
    st.dataframe(conf_pivot.astype(int))

    conf_long = conf_counts.copy()
    heat_conf = (
        alt.Chart(conf_long)
        .mark_rect()
        .encode(
            x=alt.X("interest_label:N", title="Interest (Yes/No)"),
            y=alt.Y("sim_label:N", title="Similarity (High/Low)"),
            color=alt.Color(
                "count:Q",
                title="Count",
                scale=alt.Scale(scheme="orangered"),  # warmer than the others
            ),
            tooltip=["sim_label", "interest_label", "count"],
        )
        .properties(height=300)
    )
    st.altair_chart(heat_conf, use_container_width=True)

    st.markdown("---")

    # Distribution of Similarity Scores (Yes vs No)
    st.markdown("#### Distribution of Similarity Scores (Yes vs No)")

    hist_chart = (
        alt.Chart(df)
        .transform_filter(alt.datum.similarity != None)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("similarity:Q", bin=alt.Bin(maxbins=20), title="Similarity score"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color(
                "answer_yes_no:N",
                title="Answer",
                scale=alt.Scale(
                    domain=["Yes", "No"],
                    range=["#1b9e77", "#d95f02"],  # green vs orange
                ),
            ),
            tooltip=["answer_yes_no", "count()"],
        )
        .properties(height=400)
    )
    st.altair_chart(hist_chart, use_container_width=True)

    st.markdown("---")

    # Similarity vs Confidence
    st.markdown("#### Similarity vs Confidence")

    scatter = (
        alt.Chart(filtered)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X("similarity:Q", title="Similarity score"),
            y=alt.Y("confidence_percent:Q", title="Confidence (%)"),
            color=alt.Color(
                "answer_yes_no:N",
                title="Answer",
                scale=alt.Scale(
                    domain=["Yes", "No"],
                    range=["#4daf4a", "#e41a1c"],  # green vs red
                ),
            ),
            size=alt.Size(
                "confidence_percent:Q", title="Bubble size ~ confidence"
            ),
            tooltip=[
                "student_id",
                "course",
                "answer_yes_no",
                alt.Tooltip("similarity:Q", format=".2f"),
                alt.Tooltip("confidence_percent:Q", format=".1f"),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("---")

    # Interest Funnel per Program
    st.markdown("#### Interest Funnel per Program")

    funnel_df = filtered.copy()
    funnel_df["funnel_category"] = np.where(
        (funnel_df["answer_yes_no"] == "Yes")
        & (funnel_df["confidence_percent"] >= conf_threshold),
        "High-confidence Yes",
        np.where(
            (funnel_df["answer_yes_no"] == "Yes"),
            "Low-confidence Yes",
            "No",
        ),
    )

    funnel_counts = (
        funnel_df.groupby(["course", "funnel_category"])
        .size()
        .reset_index(name="count")
    )

    funnel_chart = (
        alt.Chart(funnel_counts)
        .mark_bar()
        .encode(
            x=alt.X("course:N", sort="-y", title="Program"),
            y=alt.Y("count:Q", title="Number of responses"),
            color=alt.Color("funnel_category:N", title="Category"),
            tooltip=["course", "funnel_category", "count"],
        )
        .properties(height=400)
    )
    st.altair_chart(funnel_chart, use_container_width=True)

    st.markdown("---")

    # Background → Interests → Program (Yes only) Sankey Diagram
    st.markdown("#### Background → Interests → Program (Yes only) Sankey Diagram")

    yes_df_full = df[df["answer_yes_no"] == "Yes"].copy()
    if (
        not yes_df_full.empty
        and "student_background" in yes_df_full.columns
        and "student_academic_interests" in yes_df_full.columns
    ):
        yes_df_full["bg_cat"] = yes_df_full["student_background"].apply(
            lambda x: simplify_category(x, "Background")
        )
        yes_df_full["interest_cat"] = yes_df_full[
            "student_academic_interests"
        ].apply(lambda x: simplify_category(x, "Interests"))

        bg_nodes = sorted(yes_df_full["bg_cat"].unique().tolist())
        int_nodes = sorted(yes_df_full["interest_cat"].unique().tolist())
        prog_nodes = sorted(yes_df_full["course"].unique().tolist())

        node_labels = bg_nodes + int_nodes + prog_nodes

        def node_index(label):
            return node_labels.index(label)

        links_source = []
        links_target = []
        links_value = []

        # Background -> Interest
        bg_int_counts = (
            yes_df_full.groupby(["bg_cat", "interest_cat"])
            .size()
            .reset_index(name="count")
        )
        for _, row in bg_int_counts.iterrows():
            links_source.append(node_index(row["bg_cat"]))
            links_target.append(node_index(row["interest_cat"]))
            links_value.append(row["count"])

        # Interest -> Program
        int_prog_counts = (
            yes_df_full.groupby(["interest_cat", "course"])
            .size()
            .reset_index(name="count")
        )
        for _, row in int_prog_counts.iterrows():
            links_source.append(node_index(row["interest_cat"]))
            links_target.append(node_index(row["course"]))
            links_value.append(row["count"])

        sankey_fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_labels,
                    ),
                    link=dict(
                        source=links_source,
                        target=links_target,
                        value=links_value,
                    ),
                )
            ]
        )

        sankey_fig.update_layout(
            height=600,
            margin=dict(l=10, r=10, t=10, b=10),
        )

        st.plotly_chart(sankey_fig, use_container_width=True)
    else:
        st.info(
            "Not enough Yes responses or missing background/interest columns to build a Sankey diagram."
        )

    st.markdown("---")

    # Student–program interest heatmap
    st.markdown("#### Student–Program Interest Heatmap (Yes rate)")

    filtered["is_yes"] = (filtered["answer_yes_no"] == "Yes").astype(int)
    heat_data = (
        filtered.groupby(["student_id", "course"])["is_yes"]
        .mean()
        .reset_index()
    )

    heatmap = (
        alt.Chart(heat_data)
        .mark_rect()
        .encode(
            x=alt.X("course:N", title="Program"),
            y=alt.Y("student_id:N", title="Student"),
            color=alt.Color(
                "is_yes:Q",
                title="Yes rate",
                scale=alt.Scale(scheme="redyellowgreen"),
            ),
            tooltip=[
                "student_id",
                "course",
                alt.Tooltip("is_yes:Q", format=".2f", title="Yes rate"),
            ],
        )
        .properties(height=500)
    )
    st.altair_chart(heatmap, use_container_width=True)

        # --- Recommendations / Next Steps for Audience ---
        # --- Recommendations / Next Steps for Audience ---
    st.markdown("---")
    st.markdown("### 🎯 Recommendations & Next Steps")

    st.markdown(
        """
**1. Repair “almost-fit” programs (from the Confusion Matrix)**

**Goal:** Turn “High similarity + No” into future Yes responses.


**2.We can build a high-conversion outreach list (from the Interest Funnel)**

**Goal:** Can derive a concrete list of “ready to convert” students.


---

**3. Clarify value where students are unsure (Similarity vs Confidence)**

**Goal:** Increase confidence for students who appear to fit but don’t feel sure yet.

---

**4. Design persona-based campaigns using the Heatmap**

**Goal:** Create 2–3 clear audience segments and a campaign plan for each.
"""
    )


# ---------------------------------
# 3) Rational Analysis Tab
# ---------------------------------
with tab_rational:
    st.subheader("🧾 Rational Analysis (Qwen Model)")

    if "rationale" not in df.columns:
        st.info("No 'rationale' column found in the data.")
    else:
        rationales = df["rationale"]
        token_lists = preprocess_rationales(rationales)

        # Keyword frequencies (unigrams)
        st.markdown("#### Keyword frequencies in rationales (unigrams)")
        unigram_df = ngram_counts(token_lists, n=1, top_n=20)

        if not unigram_df.empty:
            unigram_chart = (
                alt.Chart(unigram_df)
                .mark_bar()
                .encode(
                    x=alt.X("ngram:N", sort="-y", title="Keyword"),
                    y=alt.Y("count:Q", title="Frequency"),
                    tooltip=["ngram", "count"],
                )
                .properties(height=400)
            )
            st.altair_chart(unigram_chart, use_container_width=True)
        else:
            st.info("Not enough text in rationales to compute keyword frequencies.")

        st.markdown("---")

        col_bi, col_tri = st.columns(2)

        # 2-grams
        with col_bi:
            st.markdown("#### Top 2-grams")
            bigram_df = ngram_counts(token_lists, n=2, top_n=20)
            if not bigram_df.empty:
                bigram_chart = (
                    alt.Chart(bigram_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("ngram:N", sort="-y", title="2-gram"),
                        y=alt.Y("count:Q", title="Frequency"),
                        tooltip=["ngram", "count"],
                    )
                    .properties(height=400)
                )
                st.altair_chart(bigram_chart, use_container_width=True)
            else:
                st.info("Not enough text to compute 2-grams.")

        # 3-grams
        with col_tri:
            st.markdown("#### Top 3-grams")
            trigram_df = ngram_counts(token_lists, n=3, top_n=20)
            if not trigram_df.empty:
                trigram_chart = (
                    alt.Chart(trigram_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("ngram:N", sort="-y", title="3-gram"),
                        y=alt.Y("count:Q", title="Frequency"),
                        tooltip=["ngram", "count"],
                    )
                    .properties(height=400)
                )
                st.altair_chart(trigram_chart, use_container_width=True)
            else:
                st.info("Not enough text to compute 3-grams.")

        st.markdown("---")

        # Search bar for rationales
        st.markdown("#### Search rationales")
        search_term = st.text_input(
            "Enter a keyword or phrase to search for in rationales",
            value="",
            placeholder="e.g., research, flexible, AI, healthcare...",
        )

        if search_term:
            mask_search = df["rationale"].fillna("").str.contains(
                search_term, case=False, na=False
            )
            search_results = df[mask_search].copy()
            st.write(f"Found {len(search_results)} matching responses.")
            show_cols = [
                "student_id",
                "course",
                "answer_yes_no",
                "confidence_percent",
                "rationale",
            ]
            show_cols = [c for c in show_cols if c in search_results.columns]
            st.dataframe(
                search_results[show_cols].reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.info("Enter a keyword above to search within rationales.")
