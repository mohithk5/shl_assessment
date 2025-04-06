# app.py
import streamlit as st
from recommendation_engine import scrape_url, prepare_input, get_recommendations,traced_get_recommendations
from evaluate import evaluate
import json

st.title("SHL Assessment Recommender")

query = st.text_area("Enter job query")
duration = st.number_input("Max assessment duration (minutes)", min_value=5, max_value=120, value=40)
top_k = st.number_input("Number of result required", min_value=3, max_value=15, value=10)
url = st.text_input("Optional Job Description URL")

if st.button("Recommend Assessments"):
    jd_text = scrape_url(url) if url else ""
    query_text = prepare_input(query, duration, jd_text)
    recommendations = traced_get_recommendations(query_text, top_k=10, max_duration=duration)
    st.write("Query Input:", query_text)
    st.subheader("Top Recommendations")
    st.table(recommendations)

st.header("🔍 Evaluation")

eval_json = st.text_area("Enter test queries as JSON array", height=300, value="""[
  {
    "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
    "duration": 40,
    "url": "",
    "relevant_assessments": ["Java Programming Test", "Team Collaboration Test"]
  }
]""")

if st.button("Run Evaluation"):
    try:
        test_queries = json.loads(eval_json)
        evaluate(test_queries, k=3)
    except Exception as e:
        st.error(f"Error parsing input or running evaluation: {e}")
    
    
