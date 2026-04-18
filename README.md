# EduPath AI: Agentic Course Recommendation & Analytics Ecosystem

An end-to-end analytical platform that uses Large Language Models (LLMs) to simulate graduate student personas and provide deep insights into academic program interest and conversion potential.

## Overview
EduPath AI bridges the gap between raw AI generations and actionable academic strategy. Using a **Qwen-7B model**, the system evaluates student profiles against course catalogs to predict interest. It then visualizes the "why" behind these decisions through three specialized Streamlit dashboards.

## Technical Stack
- **AI/ML:** Qwen-7B (LLM), RAG (Retrieval-Augmented Generation)
- **Frontend:** Streamlit (Multi-dashboard architecture)
- **Data Analysis:** Pandas, NumPy, Scikit-Learn (Text Vectorization)
- **Visualization:** Altair, Plotly, Vega-Lite
- **Environment:** Docker, Dev Containers

## Dashboard Modules

### 1. Executive Interest Dashboard (`application.py`)
Focuses on high-level conversion metrics for academic recruiters.
- **Interest Funnel:** Segments students into High-Confidence and Low-Confidence "Yes" groups.
- **Sankey Diagram:** Maps the journey from Student Background → Academic Interests → Program Recommendations.
- **Conversion Metrics:** Real-time tracking of "Yes" rates and program popularity.

### 2. Technical Analytics Hub (`app.py`)
Designed for data scientists to validate model performance.
- **Similarity vs. Confidence:** Scatter plots to identify where the model is "unsure."
- **N-Gram Analysis:** Distribution of unigrams and bigrams within AI-generated rationales.
- **Heatmaps:** Student × Course matrix based on similarity scores.

### 3. NLP Rationale Engine (`rationale_analytics.py`)
Deep dive into the linguistic patterns of the AI's reasoning.
- **Comparative NLP:** Direct keyword overlap analysis between "Yes" and "No" rationales.
- **Multi-Gram Frequency:** Analysis from unigrams up to 5-grams to capture complex academic phrases.


# Rationale Analytics Dashboard (Qwen)

Run locally:
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt
3) streamlit run app/rationale_analytics_app.py

Data:
- Place `outputs/recommendations.csv` in the repo
- Or provide your own CSV path in the app sidebar
