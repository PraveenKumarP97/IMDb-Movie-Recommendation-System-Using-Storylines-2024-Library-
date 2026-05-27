import streamlit as st
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IMDb Movie Recommender", 
    layout="wide", 
    page_icon="🎬"
)

# --- 2. TEXT CLEANING FUNCTION ---
# Must match your preprocessing logic exactly
def clean_user_input(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# --- 3. CACHED ASSET LOADING ---
# Optimization: This keeps the app fast by loading files into memory only once
@st.cache_resource
def load_pipeline_assets():
    try:
        df = pd.read_csv("processed_movies.csv")
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("tfidf_matrix.pkl", "rb") as f:
            matrix = pickle.load(f)
        return df, vectorizer, matrix
    except FileNotFoundError:
        return None, None, None

df, tfidf, tfidf_matrix = load_pipeline_assets()

# --- 4. USER INTERFACE LAYOUT ---
st.title("🎬 2024 IMDb Movie Recommendation System")
st.markdown("Discover movies based on plot concepts and semantic meanings using Natural Language Processing.")
st.markdown("---")

if df is None:
    st.error("⚠️ Pipeline assets missing! Please make sure 'processed_movies.csv', 'tfidf_vectorizer.pkl', and 'tfidf_matrix.pkl' are in this directory.")
else:
    # Divide the screen into two columns
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("What's your story idea?")
        user_query = st.text_area(
            label="Type or paste a movie plot storyline:",
            placeholder="e.g., A detective uncovers dark secrets in a rainy cyberpunk city...",
            height=150
        )
        
        recommend_button = st.button("Search Similar Movies", type="primary")
        
    with col2:
        st.subheader("Top 5 Recommendations")
        
        if recommend_button:
            if user_query.strip() == "":
                st.warning("Please type a storyline to get recommendations.")
            else:
                with st.spinner("Calculating similarity vectors..."):
                    # 1. Clean the incoming text
                    cleaned_query = clean_user_input(user_query)
                    
                    # 2. Vectorize the input using the existing vector space mapping
                    query_vector = tfidf.transform([cleaned_query])
                    
                    # 3. Calculate similarity score matrix
                    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
                    
                    # 4. Filter for the top 5 match index identifiers
                    top_indices = similarity_scores.argsort()[-5:][::-1]
                    
                    # 5. Display recommendation components dynamically
                    for rank, idx in enumerate(top_indices, 1):
                        match_percentage = similarity_scores[idx] * 100
                        movie_title = df.iloc[idx]['Movie Name']
                        original_story = df.iloc[idx]['Storyline']
                        
                        # Display clean card drop-downs for each movie
                        with st.expander(f"🏅 Rank {rank}: {movie_title} ({match_percentage:.1f}% Match)"):
                            st.write(f"**Storyline:** {original_story}")