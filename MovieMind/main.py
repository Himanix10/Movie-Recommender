from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import os
from typing import Optional, Tuple, List, Dict, Any

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'fallback-secret-key')

# Global variables to store data
movies_df: Optional[pd.DataFrame] = None
ratings_df: Optional[pd.DataFrame] = None
user_item_matrix: Optional[pd.DataFrame] = None
movie_features: Optional[pd.DataFrame] = None
tfidf_matrix: Optional[Any] = None

def load_data():
    """Load and preprocess the MovieLens dataset"""
    global movies_df, ratings_df, user_item_matrix, movie_features, tfidf_matrix
    
    try:
        # Load datasets
        movies_df = pd.read_csv('ml-latest-small/movies.csv')
        ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
        
        # Create user-item matrix for collaborative filtering
        user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Prepare movie features for content-based filtering
        movies_df['genres'] = movies_df['genres'].fillna('')
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])
        
        print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        print(f"User-item matrix shape: {user_item_matrix.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    return True

def content_based_recommendations(movie_title: str, top_n: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Content-based filtering using movie genres"""
    try:
        if movies_df is None or tfidf_matrix is None:
            return [], "Data not loaded properly. Please restart the application."
            
        # Find the movie (using literal matching to prevent regex injection)
        movie_match = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False, regex=False)]
        
        if movie_match.empty:
            return [], f"Movie '{movie_title}' not found. Try a different title."
        
        # Get the first matching movie
        movie_pos = movie_match.index[0]
        movie_id = movie_match.iloc[0]['movieId']
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[movie_pos:movie_pos+1], tfidf_matrix).flatten()
        
        # Get top similar movies (excluding the input movie)
        similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
        
        recommendations = []
        for idx in similar_indices:
            movie_info = movies_df.iloc[idx]
            similarity_score = cosine_sim[idx]
            recommendations.append({
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'similarity': round(similarity_score, 3)
            })
        
        return recommendations, None
        
    except Exception as e:
        return [], f"Error in content-based filtering: {str(e)}"

def collaborative_filtering_recommendations(user_id: str, top_n: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Collaborative filtering using user-based similarity"""
    try:
        if user_item_matrix is None or movies_df is None:
            return [], "Data not loaded properly. Please restart the application."
            
        user_id_int = int(user_id)
        
        if user_id_int not in user_item_matrix.index:
            return [], f"User ID {user_id_int} not found. Try a number between 1 and {user_item_matrix.index.max()}."
        
        # Get user's rating vector
        user_ratings = user_item_matrix.loc[user_id_int].values.reshape(1, -1)
        
        # Calculate user similarities using cosine distance
        user_similarities = 1 - pairwise_distances(user_ratings, user_item_matrix.values, metric='cosine').flatten()
        
        # Find most similar users (excluding the target user)
        similar_users_idx = user_similarities.argsort()[::-1][1:50]  # Top 50 similar users
        similar_users = user_item_matrix.index[similar_users_idx]
        
        # Get movies rated by similar users but not by target user
        user_unrated = user_item_matrix.loc[user_id_int] == 0
        recommendations_scores = {}
        
        for similar_user in similar_users:
            similar_user_ratings = user_item_matrix.loc[similar_user]
            similarity_score = user_similarities[user_item_matrix.index.get_loc(similar_user)]
            
            # Consider movies rated highly (>= 4.0) by similar users
            highly_rated = similar_user_ratings >= 4.0
            candidate_movies = user_unrated & highly_rated
            
            for movie_id in user_item_matrix.columns[candidate_movies]:
                if movie_id not in recommendations_scores:
                    recommendations_scores[movie_id] = 0
                recommendations_scores[movie_id] += similarity_score * similar_user_ratings[movie_id]
        
        # Sort and get top recommendations
        if not recommendations_scores:
            return [], "No recommendations found for this user. Try a different user ID."
        
        top_movies = sorted(recommendations_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recommendations = []
        for movie_id, score in top_movies:
            movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            recommendations.append({
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'score': round(score, 3)
            })
        
        return recommendations, None
        
    except ValueError:
        return [], "Please enter a valid user ID (number)."
    except Exception as e:
        return [], f"Error in collaborative filtering: {str(e)}"

def hybrid_recommendations(movie_title: str, user_id: str, top_n: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Hybrid approach combining content-based and collaborative filtering"""
    try:
        if movies_df is None:
            return [], "Data not loaded properly. Please restart the application."
            
        # Get recommendations from both methods
        content_recs, content_error = content_based_recommendations(movie_title, top_n=10)
        collab_recs, collab_error = collaborative_filtering_recommendations(user_id, top_n=10)
        
        if content_error and collab_error:
            return [], f"Both methods failed. Content: {content_error} Collaborative: {collab_error}"
        
        # Combine recommendations with weighted scoring
        combined_scores = {}
        
        # Weight content-based recommendations (40%)
        if not content_error:
            for i, rec in enumerate(content_recs):
                title = rec['title']
                # Higher rank = higher score, normalize by position
                score = (len(content_recs) - i) / len(content_recs) * 0.4
                combined_scores[title] = combined_scores.get(title, 0) + score
        
        # Weight collaborative recommendations (60%)
        if not collab_error:
            for i, rec in enumerate(collab_recs):
                title = rec['title']
                # Higher rank = higher score, normalize by position
                score = (len(collab_recs) - i) / len(collab_recs) * 0.6
                combined_scores[title] = combined_scores.get(title, 0) + score
        
        if not combined_scores:
            return [], "No recommendations could be generated."
        
        # Sort by combined score and get top recommendations
        top_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recommendations = []
        for title, score in top_movies:
            movie_info = movies_df[movies_df['title'] == title].iloc[0]
            recommendations.append({
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'hybrid_score': round(score, 3)
            })
        
        return recommendations, None
        
    except Exception as e:
        return [], f"Error in hybrid filtering: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    
    # POST request - process recommendation
    method = request.form.get('method')
    movie_input = request.form.get('movie_title', '').strip()
    user_input = request.form.get('user_id', '').strip()
    
    recommendations = []
    error_message = None
    
    try:
        if method == 'content':
            if not movie_input:
                error_message = "Please enter a movie title for content-based filtering."
            else:
                recommendations, error_message = content_based_recommendations(movie_input)
                
        elif method == 'collaborative':
            if not user_input:
                error_message = "Please enter a user ID for collaborative filtering."
            else:
                recommendations, error_message = collaborative_filtering_recommendations(user_input)
                
        elif method == 'hybrid':
            if not movie_input or not user_input:
                error_message = "Please enter both movie title and user ID for hybrid filtering."
            else:
                recommendations, error_message = hybrid_recommendations(movie_input, user_input)
        
        else:
            error_message = "Please select a recommendation method."
            
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
    
    return render_template('index.html', 
                         recommendations=recommendations, 
                         error_message=error_message,
                         method=method,
                         movie_input=movie_input,
                         user_input=user_input)

if __name__ == '__main__':
    # Load data on startup
    if load_data():
        print("Data loaded successfully!")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load data. Please check the dataset files.")