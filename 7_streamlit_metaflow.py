from metaflow import FlowSpec, step, Parameter, card
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecWithCardFlow(FlowSpec):
    
    # Parameter to select a movie title for which recommendations are desired
    movie_title = Parameter('movie_title',
                              default='Inception',
                              help='Movie title for which recommendations are desired')
    
    @step
    def start(self):
        """
        Load and preprocess the movie dataset.
        """
        # Load the dataset (ensure the CSV file is in the specified path)
        self.df = pd.read_csv('movie-genre-from-its-poster/MovieGenre.csv', encoding='latin1')
        # Use only the top 100 rows
        self.df = self.df.head(100)
        # Replace '|' with space in the Genre column and standardize the Title format
        self.df['Genre'] = self.df['Genre'].str.replace('|', ' ')
        self.df['Title'] = self.df['Title'].str.title()
        # Keep only the required columns and drop rows with missing values
        self.df = self.df[['Title', 'Genre', 'Poster']].dropna().reset_index(drop=True)
        self.next(self.compute_similarity)
    
    @step
    def compute_similarity(self):
        """
        Compute the TF-IDF matrix and cosine similarity on the Genre column.
        """
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['Genre'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.next(self.recommend)
    
    @card(type='default')
    @step
    def recommend(self):
        """
        Generate movie recommendations for the given movie title,
        validate poster URLs, and create an HTML snippet to render as a card.
        """
        def recommend_movies(movie_title, df, cosine_sim):
            if movie_title not in df['Title'].values:
                return []
            idx = df[df['Title'] == movie_title].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            # Sort movies by similarity and ignore the first one (the movie itself)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            recommended_movies = [df.iloc[i[0]] for i in sim_scores]
            return recommended_movies

        self.recommendations = recommend_movies(self.movie_title, self.df, self.cosine_sim)
        
        # Validate the poster URLs: only include movies with valid image responses
        valid_recommendations = []
        for movie in self.recommendations:
            try:
                response = requests.get(str(movie['Poster']), stream=True, timeout=5)
                if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                    valid_recommendations.append(movie)
            except Exception:
                continue
        
        # Build an HTML snippet to display the recommended movies
        if valid_recommendations:
            html = f"<h2>Recommended movies for '{self.movie_title}':</h2>"
            html += "<div style='display: flex; flex-wrap: wrap;'>"
            for movie in valid_recommendations:
                html += f"""
                <div style='flex: 1; margin: 10px; text-align: center;'>
                    <img src="{movie['Poster']}" alt="{movie['Title']}" style="width:150px;"/>
                    <p>{movie['Title']}</p>
                </div>
                """
            html += "</div>"
        else:
            html = f"<h2>No valid recommendations found for '{self.movie_title}'.</h2>"
        
        # Attach the HTML content to the card.
        # The default card in Metaflow will render the contents of `self.card_html`.
        self.card_html = html
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow.
        """
        print("Flow complete. Open the card to view the recommendations.")

if __name__ == '__main__':
    MovieRecWithCardFlow()
