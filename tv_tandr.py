import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tv=pd.read_csv('tv_shows.csv')

tv['combined']=tv['name'].astype(str)+' '+tv['overview'].astype(str)+' '+tv['genre_ids'].astype(str)+' '+tv['popularity'].astype(str)+' '+tv['vote_average'].astype(str)+' '+tv['vote_count'].astype(str)+' '+tv['adult'].astype(str)+' '+tv['poster_path'].astype(str)+' '+tv['id'].astype(str)+' '
vectorizer1=TfidfVectorizer(stop_words='english')

feature_matrix1=vectorizer1.fit_transform(tv['combined'])

def get_recommendations1(name,tv,similarity1,n=10):
    index1_list=tv.index[tv['name']==name].tolist()
    if not index1_list:
        print("Show not found.")
        return tv.iloc[[]][['name', 'overview', 'poster_path']]
    index1=index1_list[0]
    all_scores=cosine_similarity(feature_matrix1[index1],feature_matrix1).flatten()

    all_scores[index1] = -1  
    top_indices = all_scores.argsort()[::-1][:n]

    return tv[['name', 'overview', 'poster_path']].iloc[top_indices]

