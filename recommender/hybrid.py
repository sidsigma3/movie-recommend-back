import numpy as np

def hybrid_recommend(
    user_id,
    user_movie_matrix,
    user_similarity,
    item_similarity,
    ratings_df,
    user_ids,
    movie_ids,
    alpha=0.6,
    top_n=10
):
    # -------- USER INDEX --------
    if user_id not in user_ids:
        return [], "User not found"

    user_idx = list(user_ids).index(user_id)

    # -------- USER-BASED SCORE --------
    user_sim_scores = user_similarity.iloc[user_idx].values
    weighted_ratings = user_sim_scores @ user_movie_matrix.toarray()

    # -------- ITEM-BASED SCORE --------
    user_vector = user_movie_matrix[user_idx].toarray().flatten()
    item_scores = item_similarity.values @ user_vector

    # -------- HYBRID SCORE --------
    final_scores = alpha * weighted_ratings + (1 - alpha) * item_scores

    # -------- REMOVE ALREADY WATCHED --------
    watched = user_vector > 0
    final_scores[watched] = 0

    # -------- TOP N --------
    top_indices = np.argsort(final_scores)[::-1][:top_n]
    recommended_ids = [movie_ids[i] for i in top_indices]

    return recommended_ids, "Hybrid collaborative filtering"