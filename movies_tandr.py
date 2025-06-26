import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies=pd.read_csv('movies_new.csv')

movies['combined']=movies['title'].astype(str)+' '+movies['overview'].astype(str)+' '+movies['release_date'].astype(str)+' '+movies['popularity'].astype(str)+' '+movies['vote_average'].astype(str)+' '+movies['vote_count'].astype(str)+' '+movies['adult'].astype(str)+' '+movies['poster_path'].astype(str)+' '+movies['id'].astype(str)+' '
vectorizer=TfidfVectorizer(stop_words='english')

feature_matrix=vectorizer.fit_transform(movies['combined'])

def get_recommendations(title,movies,similarity,n=10):
    index_list=movies.index[movies['title']==title].tolist()
    if not index_list:
        print("Title not found.")
        return movies.iloc[[]][['title', 'overview', 'poster_path']]
    index=index_list[0]
    all_scores=cosine_similarity(feature_matrix[index],feature_matrix).flatten()

    all_scores[index] = -1  
    top_indices = all_scores.argsort()[::-1][:n]

    return movies[['title', 'overview', 'poster_path']].iloc[top_indices]
