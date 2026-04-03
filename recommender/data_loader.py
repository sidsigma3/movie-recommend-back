import pandas as pd
from scipy.sparse import csr_matrix

def load_data():
    ratings = pd.read_csv("data/ratings.csv")

    user_movie_df = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    user_movie_matrix = csr_matrix(user_movie_df.values)

    return ratings, user_movie_matrix, user_movie_df.index, user_movie_df.columns