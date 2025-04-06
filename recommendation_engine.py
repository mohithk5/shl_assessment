# recommendation_engine.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langsmith import traceable

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1",trust_remote_code=True)

catalog = pd.read_csv("data.csv")
embeddings = torch.load("embeddings.pth")

handler = ConsoleCallbackHandler() 

def scrape_url(url):
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
        return soup.get_text(separator=' ')
    except Exception as e:
        return ""
    
def clean_query_text(text):
    replacements = {
        "Java Script": "JavaScript",
        "java script": "JavaScript",
        "Java script": "JavaScript"
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

def prepare_input(query, duration, jd_text=""):
    cleaned_query = clean_query_text(query)
    input_text = f"{cleaned_query}. Candidate should complete assessment in {duration} minutes. {jd_text}"
    return input_text.strip()

def get_recommendations(query_text, top_k=10,max_duration = None):
    query_embedding = model.encode(query_text)
    scores = util.cos_sim(query_embedding, embeddings)[0].numpy()
    ranked_indices = np.argsort(-scores)

    results = []
    for idx in ranked_indices:
        item = catalog.iloc[idx]
        print(f"Matched: {item['name']} with duration {item['assessment_length']}")

        result = {
            "name": item["name"],
            "url": item["url"],
            "remote_testing": item["remote"],
            "adaptive": item["adaptive"],
            "duration": item['assessment_length'],
            "test_type": item["test_types"],
        }
        results.append(result)

        if len(results) >= top_k:
            break

    return results

@traceable(name="SHL Recommendation Trace")
def traced_get_recommendations(query_text, top_k=10, max_duration=None):
    return get_recommendations(query_text, top_k, max_duration)
    