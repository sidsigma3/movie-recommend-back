from collections import defaultdict
from recommender.data_loader import load_ratings

print("🔄 Loading recommendation model (safe mode)...")

# Load once (small memory footprint)
ratings = load_ratings()

print("✅ Core data loaded (no heavy matrices)")

def get_recommendations_from_ratings(user_ratings, top_n=10):
    """
    Render-safe collaborative filtering
    No sklearn, no matrices
    """

    if len(user_ratings) < 3:
        return [], "Not enough ratings"

    input_movie_ids = {r.movieId for r in user_ratings}

    # 1️⃣ Find users who rated the same movies
    similar_users = set()

    for _, row in ratings.iterrows():
        if row["movieId"] in input_movie_ids and row["rating"] >= 3:
            similar_users.add(row["userId"])

    # 2️⃣ Score candidate movies
    scores = defaultdict(float)
    counts = defaultdict(int)

    for _, row in ratings.iterrows():
        if row["userId"] in similar_users:
            mid = row["movieId"]
            if mid not in input_movie_ids:
                scores[mid] += row["rating"]
                counts[mid] += 1

    # 3️⃣ Rank by average rating
    ranked = sorted(
        scores.keys(),
        key=lambda m: scores[m] / counts[m],
        reverse=True
    )

    return ranked[:top_n], "Based on similar users"
