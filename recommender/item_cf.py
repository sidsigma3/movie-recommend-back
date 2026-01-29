from sklearn.metrics.pairwise import cosine_similarity

def build_item_similarity(user_movie_matrix):
    """
    Returns dense item-item similarity.
    IMPORTANT: This is computed lazily.
    """
    return cosine_similarity(user_movie_matrix.T)
