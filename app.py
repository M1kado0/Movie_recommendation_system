import streamlit as st
import faiss
import requests
import pandas as pd
import numpy as np
import yaml

movies = pd.read_csv('data/merged_data.csv')
movies_list = movies['title'].values

st.title("Movie recommendation system")
selected_movie = st.selectbox('Select a movie:', movies_list)

def fetch_posters(movie_ids):
    full_paths = []
    for movie_id in movie_ids:
        with open('configs/secrets.yaml', 'r') as f:
            configs = yaml.safe_load(f)
        API_KEY = configs['API_KEY']
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        data = requests.get(url, headers=headers)
        data = data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w1280/" + poster_path
        full_paths.append(full_path)
    return full_paths

def recommend(movie):
    input_movie = movies[movies.title == movie].iloc[0]
    index = faiss.read_index("index")

    res = requests.post('http://localhost:11434/api/embeddings',
                        json={
                            'model': 'llama2',
                            'prompt': input_movie['textual_representation']
                        })

    embedding = np.array([res.json()['embedding']], dtype='float32')
    D, I = index.search(embedding, 5)
    best_matches = np.array(movies['title'])[I.flatten()]
    best_matches_posters = fetch_posters(np.array(movies['movie_id'][I.flatten()]))
    return best_matches, best_matches_posters

if st.button("Recommend"):
    best_matches, best_matches_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    for i, col in enumerate(cols):
        with col:
            st.text(best_matches[i])
            st.image(best_matches_posters[i])
    