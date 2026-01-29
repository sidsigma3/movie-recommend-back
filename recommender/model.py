import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recommender.data_loader import load_data, load_movies
from recommender.user_cf import build_user_similarity
from recommender.item_cf import build_item_similarity
from recommender.hybrid import hybrid_recommend

print("🔄 Loading recommendation model...")

# Load data
ratings, user_movie_matrix = load_data()
movies_df = load_movies()

# Precompute similarities
user_similarity = build_user_similarity(user_movie_matrix)
item_similarity = build_item_similarity(user_movie_matrix)

print("✅ Model loaded successfully")


# ---------- HELPERS ----------

def get_movie_details(movie_ids):
    """Return movie metadata for given MovieLens movieIds"""
    filtered = movies_df[movies_df["movieId"].isin(movie_ids)]
    return filtered[["movieId", "title", "genres"]].to_dict(orient="records")


# ---------- STANDARD USER-ID RECOMMENDATION ----------

def get_recommendations(user_id, top_n=10):
    movie_ids, reason = hybrid_recommend(
        user_id,
        user_movie_matrix,
        user_similarity,
        item_similarity,
        ratings,
        alpha=0.6,
        top_n=top_n
    )

    movies = get_movie_details(movie_ids)
    return movies, reason


# ---------- SESSION / RATING-BASED RECOMMENDATION ----------

def get_recommendations_from_ratings(user_ratings, top_n=10):
    temp_user_id = -1

    temp_matrix = user_movie_matrix.copy()
    temp_matrix.loc[temp_user_id] = 0

    for r in user_ratings:
        if r.movieId in temp_matrix.columns:
            temp_matrix.at[temp_user_id, r.movieId] = r.rating

    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd

    user_sim = cosine_similarity(temp_matrix)
    user_sim_df = pd.DataFrame(
        user_sim,
        index=temp_matrix.index,
        columns=temp_matrix.index
    )

    movies = hybrid_recommend(
        temp_user_id,
        temp_matrix,
        user_sim_df,
        item_similarity,
        ratings,
        alpha=0.7,
        top_n=top_n
    )

    # 🔥 IMPORTANT: RETURN ONLY MOVIE IDS
    return movies, "Based on your selected and rated movies"
