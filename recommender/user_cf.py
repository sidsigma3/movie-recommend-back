from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_user_similarity(user_movie_matrix):
    similarity = cosine_similarity(user_movie_matrix)
    return pd.DataFrame(
        similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )

def user_cf_score(user_id, user_movie_matrix, user_similarity):
    if user_id not in user_movie_matrix.index:
        return None

    user_sim_scores = user_similarity.loc[user_id]
    scores = user_movie_matrix.T.dot(user_sim_scores)
    return scores
