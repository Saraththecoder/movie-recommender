import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_title(title):
    title = title.lower()
    title = re.sub(r"[^a-z0-9 ]", "", title)
    return title

def load_model(csv_path=r"C:\Users\sarat\OneDrive\Desktop\suraksha\movies.csv"):
    movies = pd.read_csv(csv_path)

    movies["clean_title"] = movies["title"].apply(clean_title)
    movies["clean_genres"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
        .str.lower()
    )

    movies["content"] = movies["clean_title"] + " " + movies["clean_genres"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(movies["content"])

    return movies, tfidf_matrix


def recommend(movie_title, movies, tfidf_matrix, n=10):
    movie_title = clean_title(movie_title)

    matches = movies[movies["clean_title"].str.contains(movie_title)]

    if matches.empty:
        return None

    idx = matches.index[0]

    similarity = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similarity[idx] = 0

    top_indices = similarity.argsort()[-n:][::-1]

    return movies.iloc[top_indices][["title", "genres"]]
