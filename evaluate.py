from recommendation_engine import prepare_input, scrape_url, get_recommendations
import numpy as np
import streamlit as st  

def precision_at_k(preds, relevant, k):
    preds_k = preds[:k]
    return sum([1 for p in preds_k if p in relevant]) / k

def recall_at_k(preds, relevant, k):
    preds_k = preds[:k]
    return sum([1 for p in preds_k if p in relevant]) / len(relevant)

def average_precision(preds, relevant, k):
    ap = 0
    num_relevant = 0
    for i in range(min(k, len(preds))):
        if preds[i] in relevant:
            num_relevant += 1
            ap += num_relevant / (i + 1)
    return ap / min(len(relevant), k) if relevant else 0

def clean_names(name):
    return name.replace("Java Script", "JavaScript")

def evaluate(test_queries, k=3):
    recalls, maps = [], []

    for item in test_queries:
        jd_text = scrape_url(item["url"]) if item["url"] else ""
        input_text = prepare_input(item["query"], item["duration"], jd_text)
        recommendations = get_recommendations(input_text, top_k=k)

        pred_names = [clean_names(rec["name"]) for rec in recommendations]
        gt = [clean_names(g) for g in item["relevant_assessments"]]


        r = recall_at_k(pred_names, gt, k)
        ap = average_precision(pred_names, gt, k)

        recalls.append(r)
        maps.append(ap)

        st.markdown(f"""
        **Query:** {item['query']}  
        **Recall@{k}:** {r:.3f}  
        **AP@{k}:** {ap:.3f}  
        ---
        """)

    st.success(f"📊 Mean Recall@{k}: {np.mean(recalls):.3f}")
    st.success(f"📊 MAP@{k}: {np.mean(maps):.3f}")
