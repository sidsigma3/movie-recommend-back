from sklearn.metrics.pairwise import cosine_similarity

def build_item_similarity(user_movie_matrix):
    """
    Build item-item cosine similarity matrix.
    Returns numpy array (NOT DataFrame).
    """
    return cosine_similarity(user_movie_matrix.T)
