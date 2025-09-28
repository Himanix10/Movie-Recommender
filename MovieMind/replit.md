# Movie Recommendation System

## Overview

This is a Flask-based movie recommendation system that provides personalized movie suggestions using both collaborative filtering and content-based filtering techniques. The application uses the MovieLens dataset to analyze user preferences and movie characteristics, offering recommendations through multiple algorithms including user-based collaborative filtering, item-based collaborative filtering, and genre-based content filtering.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Web Interface**: Single-page Flask application with HTML templates
- **User Interaction**: Form-based interface for selecting recommendation types and parameters
- **Styling**: CSS-based styling with gradient backgrounds and modern UI elements
- **Response Handling**: JSON-based API responses for dynamic content updates

### Backend Architecture
- **Framework**: Flask web framework with RESTful API endpoints
- **Data Processing**: Pandas for data manipulation and NumPy for numerical operations
- **Machine Learning**: Scikit-learn for recommendation algorithms including:
  - TF-IDF vectorization for content-based filtering
  - Cosine similarity for user and item comparisons
  - Pairwise distance calculations for collaborative filtering
- **Global State Management**: In-memory data storage for preprocessed datasets and matrices

### Data Architecture
- **Dataset**: MovieLens small dataset (ml-latest-small) containing:
  - Movies with titles and genres
  - User ratings (5-star scale)
  - 100,836 ratings across multiple users and movies
- **Data Structures**:
  - User-item matrix for collaborative filtering
  - TF-IDF matrix for content-based recommendations
  - Movie features dataframe for metadata handling

### Recommendation Algorithms
- **Content-Based Filtering**: Uses movie genres and TF-IDF vectorization to find similar movies
- **Collaborative Filtering**: Implements both user-based and item-based approaches using cosine similarity
- **Hybrid Approach**: Combines multiple recommendation strategies for improved accuracy

## External Dependencies

### Python Libraries
- **Flask**: Web framework for HTTP handling and template rendering
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing operations
- **Scikit-learn**: Machine learning algorithms and similarity metrics

### Data Source
- **MovieLens Dataset**: GroupLens Research dataset providing movie ratings and metadata
- **File Format**: CSV files for movies, ratings, and metadata
- **License**: Academic/research use with attribution requirements

### Environment Configuration
- **Session Management**: Configurable secret key via environment variables
- **Data Loading**: File-based data loading from local CSV files
- **Error Handling**: Exception handling for data loading and processing operations