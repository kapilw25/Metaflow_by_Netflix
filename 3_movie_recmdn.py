import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    'title': ['Inception', 'Interstellar', 'The Matrix', 'Avengers', 'The Dark Knight',
              'The Godfather', 'Pulp Fiction', 'Forrest Gump', 'Fight Club', 'The Social Network'],
    'genre': ['Sci-Fi Thriller', 'Sci-Fi Drama', 'Action Sci-Fi', 'Superhero Action', 'Action Crime',
               'Crime Drama', 'Crime Thriller', 'Drama Romance', 'Drama Thriller', 'Drama Biography']
}

# Create DataFrame
df = pd.DataFrame(data)

# Vectorize genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def recommend_movies(movie_title, df, cosine_sim):
    if movie_title not in df['title'].values:
        return f"{movie_title} not found in the dataset."
    
    idx = df[df['title'] == movie_title].index[0]  # Get index of the input movie
    sim_scores = list(enumerate(cosine_sim[idx]))  # Pair each movie with similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 similar movies
    
    recommended_movies = [df.iloc[i[0]]['title'] for i in sim_scores]
    return recommended_movies

# Example usage
movie_input = 'Inception'
print(f"Recommended movies for '{movie_input}': {recommend_movies(movie_input, df, cosine_sim)}")
