from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json

from recommender.model import get_recommendations_from_ratings

# ---------------- APP ----------------
app = FastAPI(title="Movie Recommendation API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MOVIES ----------------
with open("data/movies_catalog.json", "r", encoding="utf-8") as f:
    MOVIE_CATALOG = json.load(f)

MOVIE_MAP = {m["movieId"]: m for m in MOVIE_CATALOG}

# ---------------- MODELS ----------------
class RatingInput(BaseModel):
    movieId: int
    rating: float

class RatingRequest(BaseModel):
    ratings: List[RatingInput]

# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/movies")
def get_movies(page: int = 1, limit: int = 50):
    start = (page - 1) * limit
    end = start + limit

    return {
        "movies": MOVIE_CATALOG[start:end],
        "total": len(MOVIE_CATALOG),
        "page": page
    }

@app.post("/recommend/from-ratings")
def recommend_from_ratings(data: RatingRequest, top_n: int = 10):
    movie_ids, reason = get_recommendations_from_ratings(
        data.ratings,
        top_n
    )

    recommended_movies = [
        MOVIE_MAP[mid]
        for mid in movie_ids
        if mid in MOVIE_MAP
    ]

    return {
        "recommendations": recommended_movies,
        "reason": reason
    }
