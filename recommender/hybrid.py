def hybrid_recommend(
    user_id,
    user_movie_matrix,
    user_similarity,
    item_similarity,
    ratings,
    alpha=0.6,
    top_n=10
):
    # Cold start user
    if user_id not in user_movie_matrix.index:
        popular = (
            ratings.groupby("movieId")
            .size()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        return popular, "Popular movies (cold start)"

    user_scores = user_movie_matrix.T.dot(user_similarity.loc[user_id])
    item_scores = item_similarity.dot(user_movie_matrix.loc[user_id])

    final_scores = alpha * user_scores + (1 - alpha) * item_scores
    recommendations = (
        final_scores.sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    return recommendations, "Hybrid collaborative filtering"
