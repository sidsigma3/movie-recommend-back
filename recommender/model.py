import numpy as np
from recommender.data_loader import load_data, load_movies
from recommender.item_cf import build_item_similarity

print("🔄 Loading recommendation model (safe mode)...")

# Load lightweight data only
ratings, user_movie_matrix, movie_ids = load_data()
movies_df = load_movies()

_item_similarity = None  # LAZY LOADED

print("✅ Core data loaded (no heavy matrices yet)")


# ---------- INTERNAL HELPERS ----------

def get_item_similarity():
    global _item_similarity
    if _item_similarity is None:
        print("⚡ Computing item similarity lazily (first request only)...")
        _item_similarity = build_item_similarity(user_movie_matrix)
    return _item_similarity


def get_movie_details(movie_ids_list):
    return (
        movies_df[movies_df["movieId"].isin(movie_ids_list)]
        [["movieId", "title", "genres"]]
        .to_dict(orient="records")
    )


# ---------- SESSION / RATING BASED RECOMMENDATION ----------

def get_recommendations_from_ratings(user_ratings, top_n=10):
    """
    user_ratings = [{movieId, rating}, ...]
    """

    item_similarity = get_item_similarity()

    movie_index = {mid: i for i, mid in enumerate(movie_ids)}
    user_vector = np.zeros(len(movie_ids))

    for r in user_ratings:
        if r.movieId in movie_index:
            user_vector[movie_index[r.movieId]] = r.rating

    scores = item_similarity.dot(user_vector)

    # Remove already-rated movies
    for r in user_ratings:
        if r.movieId in movie_index:
            scores[movie_index[r.movieId]] = 0

    top_indices = np.argsort(scores)[::-1][:top_n]
    recommended_movie_ids = [movie_ids[i] for i in top_indices]

    return recommended_movie_ids, "Based on your selected and rated movies"
