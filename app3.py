import streamlit as st
import pandas as pd
from movies_tandr import get_recommendations
from tv_tandr import get_recommendations1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies=pd.read_csv('movies_new.csv')
tv=pd.read_csv('tv_shows.csv')

movies['combined'] = (
    movies['title'].astype(str) + ' ' +
    movies['overview'].astype(str) + ' ' +
    movies['release_date'].astype(str) + ' ' +
    movies['popularity'].astype(str) + ' ' +
    movies['vote_average'].astype(str) + ' ' +
    movies['vote_count'].astype(str)
)
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(movies['combined'])

tv['combined'] = (
    tv['name'].astype(str) + ' ' +
    tv['overview'].astype(str) + ' ' +
    tv['genre_ids'].astype(str) + ' ' +
    tv['popularity'].astype(str) + ' ' +
    tv['vote_average'].astype(str) + ' ' +
    tv['vote_count'].astype(str)
)
vectorizer1 = TfidfVectorizer(stop_words='english')
feature_matrix1 = vectorizer1.fit_transform(tv['combined'])


st.set_page_config(
    page_title="🎬 Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
    <style>
    .custom-heading {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
    }
    [data-baseweb="select"] {
    margin-top: -8px !important;
    }
    div[role="radiogroup"] > label {
        font-size: 1.3rem !important;
        padding: 16px 32px;
        margin: 8px 8px 8px 0;
        border-radius: 12px;
        border: 2px solid #444;
        background: #222831;
        color: #fff;
        font-weight: bold;
        transition: 0.3s;
        cursor: pointer;
    }
    div[role="radiogroup"] > label[data-selected="true"] {
        background: #fff !important;
        color: #222831 !important;
        border: 2px solid #222831;
    }
    div.stButton > button {
        font-size: 1.2rem;
        font-weight: bold;
        background-color: #222831 !important;
        color: #fff !important;
        padding: 12px 36px;
        border-radius: 12px;
        border: 2px solid #393E46;
        margin-top: 16px;
        margin-bottom: 16px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #fff;
        color: #393E46;
        border: 2px solid #FFD369;
    }
    .stSelectbox > div[data-baseweb="select"] {
        font-size: 1.1rem;
        min-height: 48px; 
    }   
    .stSlider label, .stSlider span {
        font-size: 1.1rem !important;
    } 
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    div.stButton > button {
        font-size: 1.8rem;
        font-weight: bold;
        padding: 16px 48px;
        border-radius: 16px;
        background-color: #FFD369;S
        color: #fff;
        margin-top: 20px;
        margin-bottom: 16px;
        border: 2px solid #222831;
        transition: 0.3s;
        cursor: pointer;
        width: 100%;
        display: block;
    }
    div.stButton > button:hover {
        background-color: #222831;
        color: #FFD369;
        border: 2px solid #FFD369;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie & TV Show Recommendation System")
st.markdown('<h4 style="color:#FFD369;">✨ Discover your next favorite!</h4>', unsafe_allow_html=True)

st.markdown('<div class="custom-heading" style="font-size:2.2rem;font-weight:bold;margin-bottom:0.5em;color:#fff;">What do you prefer?</div>', unsafe_allow_html=True)
option = st.radio("", ["Movies", "TV Shows"], key="big_radio")

if option=='Movies':
    st.markdown("<h4 style='color:#FFD369;'>Oh great, let's help you find some movies! 🍿</h4>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.2rem;font-weight:600;color:#fff;margin-bottom:12px;">Select a movie:</div>', unsafe_allow_html=True)
    selected_title = st.selectbox('',movies['title'].sort_values(),index=0,label_visibility='collapsed')
    n = st.slider('Number of recommendations:', 5, 20, 10)
    if st.button('Recommend Movies'):
        recs = get_recommendations(selected_title, movies,feature_matrix, n)
        if recs.empty:
            st.info("No recommendations found.")
        else:
            for _, row in recs.iterrows():
                st.markdown(f"""
                    <div style="background-color:#393E46;padding:24px;border-radius:18px;margin-bottom:18px;">
                        <h2 style="color:#FFD369;">{row['title']}</h2>
                        <img src="https://image.tmdb.org/t/p/w500{row['poster_path']}" width="180" style="border-radius:8px;">
                        <p style="color:#EEEEEE;font-size:1.1rem;">{row['overview']}</p>
                    </div>
                """, unsafe_allow_html=True)
elif option == 'TV Shows':
    st.markdown("<h4 style='color:#FFD369;'>Oh great, let's help you find some TV shows! 📺</h4>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.2rem;font-weight:600;color:#fff;margin-bottom:12px;">Select a TV show:</div>', unsafe_allow_html=True)
    selected_name = st.selectbox('',tv['name'].sort_values(),index=0,label_visibility='collapsed')
    n = st.slider('Number of recommendations:', 5, 20, 10, key='tv')
    if st.button('Recommend TV Shows'):
        recs1 = get_recommendations1(selected_name, tv, feature_matrix1, n)
        if recs1.empty:
            st.info("No recommendations found.")
        else:
            for _, row in recs1.iterrows():
                st.markdown(f"""
                    <div style="background-color:#393E46;padding:24px;border-radius:18px;margin-bottom:18px;">
                        <h2 style="color:#FFD369;">{row['name']}</h2>
                        <img src="https://image.tmdb.org/t/p/w500{row['poster_path']}" width="180" style="border-radius:8px;">
                        <p style="color:#EEEEEE;font-size:1.1rem;">{row['overview']}</p>
                    </div>
                """, unsafe_allow_html=True)
