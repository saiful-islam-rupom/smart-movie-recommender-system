import streamlit as st
import pickle
import joblib
import requests
from urllib.parse import unquote

TMDB_API_KEY = "32abef11b2a23303c53cd305c615225b"

df = pickle.load(open("df.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))
nn_model = joblib.load("nn_model_compressed.pkl")
vectors = cv.transform(df["tags"]).toarray()

# Fetch movie details
def fetch_movie_details(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return {"title": "N/A", "overview": "No description available."}
    data = response.json()
    return {
        "title": data.get("title"),
        "overview": data.get("overview"),
    }

# Fetch actor info with fallback image
def fetch_actors(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    actors = []
    default_img_url = "https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png"

    for person in data.get("cast", [])[:9]:
        name = person["name"]
        profile_path = person.get("profile_path")
        img_url = f"https://image.tmdb.org/t/p/w185{profile_path}" if profile_path else default_img_url
        actors.append((name, img_url))
    return actors

# Recommend similar movies
def recommend(movie_title):
    index = df[df["title"] == movie_title].index[0]
    movie_vector = vectors[index].reshape(1, -1)
    distances, indices = nn_model.kneighbors(movie_vector, n_neighbors=11)
    recommended_titles = []
    for idx in indices[0]:
        rec_title = df.iloc[idx]["title"]
        if rec_title != movie_title and rec_title not in recommended_titles:
            recommended_titles.append(rec_title)
        if len(recommended_titles) == 5:
            break
    rec_indices = df[df["title"].isin(recommended_titles)].index.tolist()
    return rec_indices

# Fetch movie poster
def fetch_poster(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    poster_path = data.get("poster_path")
    return f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else None

# App Logic
st.set_page_config(layout="wide")
st.title("Smart Movie Recommender System")
st.markdown("Developed by [**Saiful Islam Rupom**](https://www.linkedin.com/in/saiful-islam-rupom/)")

# Handle movie selection from URL query param
query_params = st.query_params
movie_query_param = None

if "movie" in query_params:
    raw_query = query_params["movie"]
    movie_query_param = unquote(raw_query[0] if isinstance(raw_query, list) else raw_query)
    if movie_query_param in df["title"].values:
        st.session_state.selected_movie = movie_query_param
        st.query_params.clear()  


# Set initial movie
if "selected_movie" not in st.session_state:
    default_title = "Lie with Me (2005)"
    if default_title in df["title"].values:
        st.session_state.selected_movie = default_title
    else:
        st.session_state.selected_movie = df["title"].iloc[0]

def update_selected(movie_name):
    st.session_state.selected_movie = movie_name
    st.rerun()

# Movie Dropdown
selected_movie = st.selectbox(
    "Select or type a movie:",
    options=df["title"].values,
    index=int(df[df["title"] == st.session_state.selected_movie].index[0]),
    key="movie_selector"
)
if selected_movie != st.session_state.selected_movie:
    update_selected(selected_movie)

# Movie Overview
movie_row = df[df["title"] == st.session_state.selected_movie].iloc[0]
tmdb_id = movie_row["tmdb_id"]
movie_info = fetch_movie_details(tmdb_id)
poster_url = fetch_poster(tmdb_id)

st.subheader(f"{movie_info['title']}")
st.write(movie_info["overview"])

# Poster + Actors layout structure
poster_col, actors_col = st.columns([2, 3])
if poster_url:
    poster_col.image(poster_url, use_container_width=True)
else:
    poster_col.warning("No poster available.")

actors = fetch_actors(tmdb_id)
with actors_col:
    st.write("**Actors:**")
    rows = [st.columns(3), st.columns(3), st.columns(3)]
    for i, (name, img) in enumerate(actors):
        target_col = rows[i // 3][i % 3]
        with target_col:
            st.image(img, width=100)
            st.caption(name)

# Recommendations
st.markdown("---")
st.subheader("You might also like:")
st.markdown("(Click any of these recommended movies for further recommendation.)")

rec_indices = recommend(st.session_state.selected_movie)
rec_cols = st.columns(5)
fallback_poster_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Film_camera_icon.svg/256px-Film_camera_icon.svg.png"

for col, idx in zip(rec_cols, rec_indices):
    rec_title = df.iloc[idx]["title"]
    rec_tmdb_id = df.iloc[idx]["tmdb_id"]
    rec_poster = fetch_poster(rec_tmdb_id)
    final_poster = rec_poster if rec_poster else fallback_poster_url
    fallback_text = ""
    if not rec_poster:
        fallback_text = '<div style="text-align:center; font-style: italic; font-size: small; color: gray;">No poster available.</div>'

    # Use safe internal routing with query param and same tab
    link = f"?movie={rec_title.replace(' ', '%20')}"
    with col:
        st.markdown(
            f'''
            <a href="{link}" target="_self" style="text-decoration:none; color:inherit;">
                <img src="{final_poster}" width="100%" style="border-radius: 8px;" />
                <div style="text-align: center; font-weight: bold; margin-top: 0.5em;">{rec_title}</div>
                {fallback_text}
            </a>
            ''',
            unsafe_allow_html=True
        )
