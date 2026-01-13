import streamlit as st
from recommender import load_model, recommend

st.set_page_config(
    page_title="🎬 Movie Recommender",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite one!")

@st.cache_data
def load_data():
    return load_model()

movies, tfidf_matrix = load_data()

movie_name = st.text_input(
    "Enter a movie title:",
    placeholder="Toy Story, Matrix, Titanic..."
)

if movie_name:
    with st.spinner("Finding similar movies..."):
        results = recommend(movie_name, movies, tfidf_matrix)

    if results is None:
        st.error("❌ Movie not found. Try another title.")
    else:
        st.success("✅ Recommended Movies")
        st.dataframe(results, use_container_width=True)
