from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_item_similarity(user_movie_matrix, movie_ids):
    sim = cosine_similarity(user_movie_matrix.T)

    return pd.DataFrame(sim, index=movie_ids, columns=movie_ids)