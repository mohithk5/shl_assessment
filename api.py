from fastapi import FastAPI
from pydantic import BaseModel
from recommendation_engine import scrape_url, prepare_input, get_recommendations

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    duration: int
    url: str = None  

@app.get("/")
def root():
    return {"message": "SHL Assessment Recommendation API is running."}


@app.post("/recommend")
def recommend(data: QueryRequest):
    jd_text = scrape_url(data.url) if data.url else ""
    input_text = prepare_input(data.query, data.duration, jd_text)
    recommendations = get_recommendations(input_text, top_k=10, max_duration=data.duration)
    return {"results": recommendations}
