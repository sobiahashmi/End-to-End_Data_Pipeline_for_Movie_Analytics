import streamlit as st
import pandas as pd
from joblib import load
import time

# Load the pre-trained model
model = load('imdb_pipeline_for_movie_analytics.joblib')

# Baloons Animation
st.sidebar.title("Welcome to My Streamlit App! 🎈")
# st.write("Click the button to celebrate!")
if st.sidebar.button("Celebrate!"):
    st.sidebar.balloons()

if st.sidebar.button("More Balloons!"):
    for _ in range(3):  # Show balloons 3 times
        st.sidebar.balloons()
        time.sleep(2)  # Add a short delay between balloon bursts

if st.sidebar.button("Party Mode!"):
    st.sidebar.balloons()
    time.sleep(2)
    st.sidebar.snow()  # Adds a snowfall effect after balloons

# Create a streamlit App
st.title("Movie Analytics Dashboard")

# Input fields for user to enter movie details
movie_id = st.number_input("Movie ID", min_value=1, step=1)
Genre = st.selectbox("Genre", options=['Drama', 'Comedy', 'Action', 'Horror'])
release_year = st.number_input("Release Year", min_value=1980, max_value=2025, step=1)
rating = st.number_input("Rating", min_value=0.0, max_value=10.0, step=0.1)
votes = st.number_input("Votes", min_value=0, step=1)

# label Mapping
label_mapping = {
    'Drama': 0,
    'Comedy': 1,
    'Action': 2,
    'Horror': 3
    }

Genre = label_mapping[Genre]

prediction = model.predict([[movie_id, Genre, release_year, rating, votes]])
st.subheader("Predicted Revenue (in Millions):")
st.write(f"${prediction[0]:.2f} million")