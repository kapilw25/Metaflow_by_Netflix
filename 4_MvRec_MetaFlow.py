from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommendationFlow(FlowSpec):
    
    # Define a parameter for the movie title with a default value
    movie_title = Parameter('movie_title',
                              default='Inception',
                              help='Movie title for which recommendations are desired')
    
    @step
    def start(self):
        # Sample movie dataset
        data = {
            'title': ['Inception', 'Interstellar', 'The Matrix', 'Avengers', 'The Dark Knight',
                      'The Godfather', 'Pulp Fiction', 'Forrest Gump', 'Fight Club', 'The Social Network'],
            'genre': ['Sci-Fi Thriller', 'Sci-Fi Drama', 'Action Sci-Fi', 'Superhero Action', 'Action Crime',
                      'Crime Drama', 'Crime Thriller', 'Drama Romance', 'Drama Thriller', 'Drama Biography']
        }
        self.df = pd.DataFrame(data)
        
        # Vectorize genres
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['genre'])
        
        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        self.next(self.recommend)
    
    @step
    def recommend(self):
        # Function to get movie recommendations
        def recommend_movies(movie_title, df, cosine_sim):
            if movie_title not in df['title'].values:
                return f"{movie_title} not found in the dataset."
            idx = df[df['title'] == movie_title].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            # Get top 5 similar movies (ignoring the input movie itself)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            recommended_movies = [df.iloc[i[0]]['title'] for i in sim_scores]
            return recommended_movies

        # Generate recommendations for the given movie_title
        self.recommendations = recommend_movies(self.movie_title, self.df, self.cosine_sim)
        print(f"Recommended movies for '{self.movie_title}': {self.recommendations}")
        self.next(self.end)
    
    @step
    def end(self):
        print("Flow finished.")

if __name__ == '__main__':
    MovieRecommendationFlow()
