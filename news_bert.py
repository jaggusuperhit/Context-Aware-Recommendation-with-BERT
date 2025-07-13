import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import torch
import os

# Set up Streamlit page config
st.set_page_config(page_title="News Recommendation System", layout="wide")

# Download stopwords and punkt (for tokenization)
try:
    nltk.download('stopwords')
    nltk.download('punkt')
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")
    st.stop()

# Initialize the PorterStemmer and stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing function to clean text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    words = text.split()
    # Remove stopwords and apply stemming
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    # Join the words back into a string
    return " ".join(processed_words)

# Function to load data with error handling
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('Articles.csv', encoding='ISO-8859-1')
        return df
    except Exception as e:
        st.error(f"Error loading articles data: {e}")
        return None

# Function to load model with error handling
@st.cache_resource
def load_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load embeddings with error handling
@st.cache_resource
def load_embeddings():
    try:
        if not os.path.exists('embeddings.pkl'):
            st.error("Embeddings file not found. Please ensure 'embeddings.pkl' is in the correct directory.")
            return None
        
        embeddings = torch.load('embeddings.pkl', map_location='cpu')
        return embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

# Function to recommend articles
def recommend_articles(query, df, model, embeddings, num_recommendations=5):
    if not query or not isinstance(query, str):
        return pd.DataFrame()
    
    # Preprocess query
    processed_query = preprocess_text(query)
    
    # Get query embedding
    query_embedding = model.encode(processed_query, convert_to_tensor=False)
    
    # Compute similarities
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Get top indices
    top_indices = similarities.argsort()[-num_recommendations:][::-1]
    
    # Return recommended articles
    return df.iloc[top_indices][['Heading', 'NewsType', 'Article', 'Date']]

# Main app function
def main():
    st.title("📰 Context-Aware News Recommendation System")
    st.markdown("Find similar news articles based on your search query")
    
    # Load resources
    with st.spinner("Loading resources..."):
        df = load_data()
        model = load_model()
        embeddings = load_embeddings()
        
        if df is None or model is None or embeddings is None:
            st.error("Failed to load required resources. Please check the error messages above.")
            return
    
    # Search interface
    st.subheader("🔍 Search for News Articles")
    query = st.text_input("Enter a news headline or topic:", placeholder="e.g. Latest technology trends")
    
    if query:
        with st.spinner("Finding similar articles..."):
            recommendations = recommend_articles(query, df, model, embeddings)
        
        if not recommendations.empty:
            st.subheader("🎯 Recommended Articles")
            
            # Display each recommendation in an expandable section
            for idx, row in recommendations.iterrows():
                with st.expander(f"📌 {row['Heading']} ({row['NewsType']} - {row['Date']})"):
                    st.markdown(f"**Article:**\n\n{row['Article']}")
                    st.markdown("---")
        else:
            st.warning("No recommendations found. Try a different search query.")

if __name__ == "__main__":
    main()