import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recommender.data_loader import load_data, load_movies
from recommender.item_cf import build_item_similarity
from recommender.hybrid import hybrid_recommend

print("🔄 Loading recommendation model (optimized)...")

# Load data
ratings, user_movie_matrix, movie_ids = load_data()
movies_df = load_movies()

# Precompute item similarity ONLY (huge memory saver)
item_similarity = build_item_similarity(user_movie_matrix)

print("✅ Model loaded successfully")


# ---------- HELPERS ----------

def get_movie_details(movie_ids):
    return (
        movies_df[movies_df["movieId"].isin(movie_ids)]
        [["movieId", "title", "genres"]]
        .to_dict(orient="records")
    )


# ---------- STANDARD USER-ID RECOMMENDATION ----------

def get_recommendations(user_id, top_n=10):
    movie_ids, reason = hybrid_recommend(
        user_id=user_id,
        user_movie_matrix=user_movie_matrix,
        item_similarity=item_similarity,
        ratings=ratings,
        alpha=0.6,
        top_n=top_n
    )

    return get_movie_details(movie_ids), reason


# ---------- SESSION / RATING-BASED RECOMMENDATION ----------

def get_recommendations_from_ratings(user_ratings, top_n=10):
    """
    user_ratings = [{movieId, rating}, ...]
    """

    # Build temp user vector aligned with movie_ids
    user_vector = np.zeros(len(movie_ids))

    movie_index_map = {mid: idx for idx, mid in enumerate(movie_ids)}

    for r in user_ratings:
        if r.movieId in movie_index_map:
            user_vector[movie_index_map[r.movieId]] = r.rating

    # Compute similarity against items
    scores = item_similarity.dot(user_vector)

    # Remove already rated movies
    for r in user_ratings:
        if r.movieId in movie_index_map:
            scores[movie_index_map[r.movieId]] = 0

    top_indices = np.argsort(scores)[::-1][:top_n]
    recommended_movie_ids = [movie_ids[i] for i in top_indices]

    return recommended_movie_ids, "Based on your selected and rated movies"
