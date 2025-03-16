from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommendationFlow(FlowSpec):
    
    # Define a parameter for a comma-separated list of movie titles
    movie_titles = Parameter('movie_titles',
                               default='Inception,Interstellar,The Matrix',
                               help='Comma separated list of movie titles for which recommendations are desired')
    
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
        
        # Convert the comma separated movie_titles parameter into a list
        self.movies = [title.strip() for title in self.movie_titles.split(',')]
        
        # Branch dynamically: one branch per movie title
        self.next(self.recommend, foreach='movies')
    
    @step
    def recommend(self):
        # self.input holds the current movie title for this branch
        current_title = self.input
        
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
        
        # Compute recommendations for the current branch movie title
        self.movie_title = current_title
        self.recommendations = recommend_movies(current_title, self.df, self.cosine_sim)
        print(f"Recommended movies for '{current_title}': {self.recommendations}")
        
        self.next(self.join)
    
    @step
    def join(self, inputs):
        # Collect recommendations from all branches
        self.all_recommendations = {}
        for branch in inputs:
            movie = branch.movie_title
            self.all_recommendations[movie] = branch.recommendations
        self.next(self.end)
    
    @step
    def end(self):
        print("Final recommendations:")
        for movie, recs in self.all_recommendations.items():
            print(f"{movie}: {recs}")
        print("Flow finished.")

if __name__ == '__main__':
    MovieRecommendationFlow()
