import pandas as pd
import json

# ---------------- CONFIG ----------------
MOVIES_CSV_PATH = "../data/movies.csv"
IMAGE_MAP_PATH = "../data/movie_poster.csv"
OUTPUT_JSON_PATH = "../data/movies_catalog.json"

MAX_MOVIES = 3500
# ----------------------------------------


def fix_amazon_image_url(url: str | None):
    """
    Fix broken Amazon image URLs like '@..jpg'
    """
    if not url or url == "nan":
        return None

    url = url.strip()

    if url.endswith("@..jpg"):
        return url.replace("@..jpg", "@._V1_.jpg")

    return url


def main():
    print("🎬 Loading MovieLens movies.csv...")
    movies_df = pd.read_csv(MOVIES_CSV_PATH)

    print("🖼️ Loading movie image map...")

    images_df = pd.read_csv(
        IMAGE_MAP_PATH,
        header=None,
        names=["movieId", "poster"],
        sep=",",
        engine="python"
    )

    # ---- CLEAN IMAGE DATA ----
    images_df["movieId"] = pd.to_numeric(images_df["movieId"], errors="coerce")
    images_df = images_df.dropna(subset=["movieId"])
    images_df["movieId"] = images_df["movieId"].astype(int)

    images_df["poster"] = images_df["poster"].astype(str)
    images_df["poster"] = images_df["poster"].apply(fix_amazon_image_url)

    print(f"🖼️ Valid image rows: {len(images_df)}")

    # ---- CLEAN MOVIE DATA ----
    movies_df["movieId"] = movies_df["movieId"].astype(int)

    print("🔗 Merging movie metadata with images...")
    merged_df = movies_df.merge(images_df, on="movieId", how="inner")

    merged_df = merged_df.head(MAX_MOVIES)

    print(f"✅ Total movies in catalog: {len(merged_df)}")

    # ---- BUILD CATALOG ----
    catalog = []
    for _, row in merged_df.iterrows():
        catalog.append({
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
            "poster": row["poster"] or "/placeholder.svg"
        })

    print("💾 Saving unified movie catalog...")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"🎉 Done! Catalog saved to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
