import pandas as pd

def load_data():
    ratings = pd.read_csv("data/ratings.csv")
    
    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    return ratings, user_movie_matrix


def load_movies():
    movies = pd.read_csv("data/movies.csv")
    return movies
