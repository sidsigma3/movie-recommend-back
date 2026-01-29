from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_item_similarity(user_movie_matrix):
    similarity = cosine_similarity(user_movie_matrix.T)
    return pd.DataFrame(
        similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns
    )

def item_cf_score(user_id, user_movie_matrix, item_similarity):
    if user_id not in user_movie_matrix.index:
        return None

    user_ratings = user_movie_matrix.loc[user_id]
    scores = item_similarity.dot(user_ratings)
    return scores
