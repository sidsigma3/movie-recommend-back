from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from recommender.model import get_recommendations
from recommender.model import get_recommendations_from_ratings
import json

app = FastAPI(title="Movie Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("data/movies_catalog.json", "r", encoding="utf-8") as f:
    MOVIE_CATALOG = json.load(f)


@app.get("/movies")
def get_movies(page: int = 1, limit: int = 50):
    start = (page - 1) * limit
    end = start + limit

    return {
        "page": page,
        "limit": limit,
        "total": len(MOVIE_CATALOG),
        "movies": MOVIE_CATALOG[start:end]
    }


@app.get("/")
def root():
    return {"status": "Movie Recommendation API is running"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):
    movies, reason = get_recommendations(user_id, top_n)
    return {
        "userId": user_id,
        "recommendations": movies,
        "reason": reason
    }


class RatingInput(BaseModel):
    movieId: int
    rating: float

class RatingRequest(BaseModel):
    ratings: List[RatingInput]

@app.post("/recommend/from-ratings")
def recommend_from_ratings(data: RatingRequest, top_n: int = 10):
    movie_ids, reason = get_recommendations_from_ratings(
        data.ratings, top_n
    )

    movie_map = {m["movieId"]: m for m in MOVIE_CATALOG}

    recommended_movies = [
        movie_map[mid] for mid in movie_ids if mid in movie_map
    ]

    return {
        "recommendations": recommended_movies,
        "reason": reason
    }
