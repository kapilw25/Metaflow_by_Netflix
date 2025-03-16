import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit UI
st.set_page_config(page_title="Netflix-Themed Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'>🎬 Movie Recommendation System 🍿</h1>", unsafe_allow_html=True)

# Cache the data loading process to improve performance
@st.cache_data
def load_data():
    df = pd.read_csv('movie-genre-from-its-poster/MovieGenre.csv', encoding='latin1')
    # reduce the size to keep only top 100 rows on the dataset
    df = df.head(100)
    df['Genre'] = df['Genre'].str.replace('|', ' ')
    df['Title'] = df['Title'].str.title()
    return df[['Title', 'Genre', 'Poster']].dropna().reset_index(drop=True)

df = load_data()

# Cache the similarity computation for faster recommendations
@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Genre'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_similarity(df)

# Function to get movie recommendations
def recommend_movies(movie_title, df, cosine_sim):
    if movie_title not in df['Title'].values:
        return []

    idx = df[df['Title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended_movies = [df.iloc[i[0]] for i in sim_scores]
    return recommended_movies



movie_input = st.selectbox("Select a movie", df['Title'].values)

if st.button("Recommend Movies"):
    recommendations = recommend_movies(movie_input, df, cosine_sim)

    if recommendations:
        # Filter valid recommendations
        valid_recommendations = []
        for movie in recommendations:
            try:
                response = requests.get(str(movie['Poster']), stream=True)
                if response.status_code == 200 and 'image' in response.headers['Content-Type']:
                    valid_recommendations.append(movie)
            except:
                continue

        # Display valid images only
        if valid_recommendations:
            st.subheader(f"Recommended movies for '{movie_input}':")
            cols = st.columns(len(valid_recommendations))
            for idx, movie in enumerate(valid_recommendations):
                with cols[idx]:
                    st.image(str(movie['Poster']), caption=movie['Title'], width=150)
        else:
            st.error("No valid movie posters found.")

    else:
        st.error("Movie not found in the dataset.")

