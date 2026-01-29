import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = "data"


def load_data():
    """
    Loads ratings and builds a sparse user-movie matrix
    """
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

    user_movie_df = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0
    )

    movie_ids = user_movie_df.columns.tolist()

    user_movie_matrix = csr_matrix(user_movie_df.values)

    return ratings, user_movie_matrix, movie_ids


def load_movies():
    """
    Loads MovieLens movies metadata
    """
    return pd.read_csv(f"{DATA_DIR}/movies.csv")
