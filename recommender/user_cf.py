from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_user_similarity(user_movie_matrix, user_ids):
    sim = cosine_similarity(user_movie_matrix)

    return pd.DataFrame(sim, index=user_ids, columns=user_ids)