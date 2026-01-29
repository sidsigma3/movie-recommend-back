import pandas as pd

RATINGS_PATH = "data/ratings.csv"

def load_ratings():
    """
    Load ratings once at startup
    Columns expected: userId, movieId, rating
    """
    df = pd.read_csv(RATINGS_PATH)
    df = df[["userId", "movieId", "rating"]]
    return df
