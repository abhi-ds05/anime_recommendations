from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import h5py
import requests

app = Flask(__name__)

def extract_weights(file_path, layer_name):
    with h5py.File(file_path, 'r') as h5_file:
        if layer_name in h5_file:
            weight_layer = h5_file[layer_name]
            if isinstance(weight_layer, h5py.Dataset):
                weights = weight_layer[()]
                weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
                return [weights]
    raise KeyError(f"Unable to find weights for layer '{layer_name}' in the HDF5 file.")

# Load your models
file_path = 'model/myanimeweights.h5'
anime_weights = extract_weights(file_path, 'anime_embedding/anime_embedding/embeddings:0')[0]
user_weights = extract_weights(file_path, 'user_embedding/user_embedding/embeddings:0')[0]

with open('model/anime_encoder.pkl', 'rb') as file:
    anime_encoder = pickle.load(file)

with open('model/user_encoder.pkl', 'rb') as file:
    user_encoder = pickle.load(file)

with open('model/anime-dataset-2023.pkl', 'rb') as file:
    df_anime = pickle.load(file)
df_anime = df_anime.replace("UNKNOWN", "")

df = pd.read_csv('model/users-score-2023.csv', low_memory=True)

# Existing home route
@app.route('/')
def home():
    return render_template('index.html')

# ===================
# Core recommendation functions
# ===================
def find_similar_users(item_input, n=10, return_dist=False, neg=False):
    try:
        index = item_input
        encoded_index = user_encoder.transform([index])[0]
        weights = user_weights
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n = n + 1
        closest = sorted_dists[:n] if neg else sorted_dists[-n:]
        SimilarityArr = []
        for close in closest:
            similarity = dists[close]
            decoded_id = user_encoder.inverse_transform([close])[0]
            SimilarityArr.append({"similar_users": decoded_id, "similarity": similarity})
        return pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
    except Exception as e:
        print(f"Error finding similar users: {e}")
        return pd.DataFrame()

def get_user_preferences(user_id):
    animes_watched = df[df['user_id'] == user_id]
    if animes_watched.empty:
        return pd.DataFrame()
    threshold = np.percentile(animes_watched.rating, 75)
    top_animes = animes_watched[animes_watched.rating >= threshold].anime_id.values
    return df_anime[df_anime["anime_id"].isin(top_animes)][["Name", "Genres"]]

def get_recommended_animes(similar_users, user_pref, n=10):
    recommended = []
    anime_list = []
    for uid in similar_users.similar_users.values:
        prefs = get_user_preferences(int(uid))
        if not prefs.empty:
            prefs = prefs[~prefs["Name"].isin(user_pref["Name"].values)]
            anime_list.append(prefs.Name.values)
    if not anime_list:
        return pd.DataFrame()
    anime_df = pd.DataFrame(anime_list)
    sorted_list = pd.DataFrame(pd.Series(anime_df.values.ravel()).value_counts()).head(n)
    count_map = df['anime_id'].value_counts()
    for anime_name in sorted_list.index:
        try:
            a_row = df_anime[df_anime.Name == anime_name].iloc[0]
            anime_id = a_row.anime_id
            n_user_pref = count_map.get(anime_id, 0)
            recommended.append({
                "anime_id": anime_id,
                "name": a_row.Name,
                "img": a_row["Image URL"],
                "genres": a_row.Genres,
                "score": a_row.Score,
                "synopsis": a_row.Synopsis,
                "users_watched": n_user_pref
            })
        except:
            pass
    return pd.DataFrame(recommended)

def find_similar_animes(name, n=10, return_dist=False, neg=False):
    try:
        a_row = df_anime[df_anime['Name'] == name].iloc[0]
        encoded_index = anime_encoder.transform([a_row['anime_id']])[0]
        weights = anime_weights
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n = n + 1
        closest = sorted_dists[:n] if neg else sorted_dists[-n:]
        SimilarityArr = []
        for close in closest:
            decoded_id = anime_encoder.inverse_transform([close])[0]
            anime = df_anime[df_anime['anime_id'] == decoded_id].iloc[0]
            SimilarityArr.append({
                "anime_id": anime.anime_id,
                "name": anime.Name,
                "img": anime["Image URL"],
                "genres": anime.Genres,
                "score": anime.Score,
                "synopsis": anime.Synopsis
            })
        return pd.DataFrame(SimilarityArr).sort_values(by="score", ascending=False)
    except Exception as e:
        print(f"Anime not found: {e}")
        return pd.DataFrame()

# ===================
# New JSON API routes
# ===================
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    recommendation_type = data.get('recommendation_type')
    num_recommendations = int(data.get('num_recommendations', 10))

    if recommendation_type == "user_based":
        user_id = int(data.get('user_id'))
        similar_users = find_similar_users(user_id, n=15, neg=False)
        similar_users = similar_users[similar_users.similarity > 0.4]
        similar_users = similar_users[similar_users.similar_users != user_id]
        user_pref = get_user_preferences(user_id)
        recs = get_recommended_animes(similar_users, user_pref, n=num_recommendations)
        return jsonify(recs.to_dict(orient='records'))

    elif recommendation_type == "item_based":
        anime_name = data.get('anime_name')
        recs = find_similar_animes(anime_name, n=num_recommendations)
        return jsonify(recs.to_dict(orient='records'))

    return jsonify({"error": "Invalid request"}), 400

# Existing autocomplete
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search_term = request.args.get('term')
    anime_names = []
    if search_term:
        filtered = df_anime[df_anime['Name'].str.contains(search_term, case=False)]
        anime_names = filtered['Name'].tolist()
    return jsonify(anime_names)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
