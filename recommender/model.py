import numpy as np
import pandas as pd

from recommender.data_loader import load_data
from recommender.user_cf import build_user_similarity
from recommender.item_cf import build_item_similarity
from recommender.hybrid import hybrid_recommend

print("🔄 Loading recommendation model...")

ratings, user_movie_matrix, user_ids, movie_ids = load_data()

user_similarity = build_user_similarity(user_movie_matrix, user_ids)
item_similarity = build_item_similarity(user_movie_matrix, movie_ids)

print("✅ Model loaded successfully")


# -------- USER ID BASED --------
def get_recommendations(user_id, top_n=10):
    movie_ids_list, reason = hybrid_recommend(
        user_id,
        user_movie_matrix,
        user_similarity,
        item_similarity,
        ratings,
        user_ids,
        movie_ids,
        alpha=0.6,
        top_n=top_n
    )

    return movie_ids_list, reason


# -------- SESSION BASED --------
def get_recommendations_from_ratings(user_ratings, top_n=10):
    temp_user_id = -1

    # Convert sparse to dense DataFrame
    df = pd.DataFrame(
        user_movie_matrix.toarray(),
        index=user_ids,
        columns=movie_ids
    )

    df.loc[temp_user_id] = 0

    # Fill user ratings
    for r in user_ratings:
        if r.movieId in df.columns:
            df.at[temp_user_id, r.movieId] = r.rating

    # Convert back to matrix
    temp_matrix = df.values

    # Recompute similarity for temp user
    from sklearn.metrics.pairwise import cosine_similarity

    user_sim = cosine_similarity(temp_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=df.index, columns=df.index)

    # Use same item similarity
    temp_user_idx = list(df.index).index(temp_user_id)

    user_vector = temp_matrix[temp_user_idx]

    user_scores = user_sim_df.iloc[temp_user_idx].values @ temp_matrix
    item_scores = item_similarity.values @ user_vector

    final_scores = 0.7 * user_scores + 0.3 * item_scores

    # Remove already rated
    final_scores[user_vector > 0] = 0

    top_indices = np.argsort(final_scores)[::-1][:top_n]
    recommended_ids = [movie_ids[i] for i in top_indices]

    return recommended_ids, "Hybrid (session-based)"