import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
df = pd.read_csv('beer_profile_and_ratings.csv')
df = df[['Beer Name (Full)', 'Style', 'Astringency', 'Body','Alcohol', 'Bitter','Sweet', 'Sour', 'Salty', 'Fruits','Hoppy', 'Spices','Malty']]
df['Style'] = df['Style'].fillna('Unknown')
df[df.columns[2:]] = df[df.columns[2:]].fillna(0)

# Streamlit UI
st.title("🍺 Beer Recommender")
st.write("Adjust the sliders to match your flavor preferences, and get beer recommendations!")

# Get flavor preferences
flavor_weights = {}
for attr in df.columns[2:]:
    flavor_weights[attr] = st.slider(attr, 0.0, 1.0, 0.5, 0.1)

# Normalize and compute similarity
scaler = MinMaxScaler()
flavor_attributes = df[df.columns[2:]]
scaled_flavor_attributes = scaler.fit_transform(flavor_attributes)

user_vector = np.array([flavor_weights[attr] for attr in flavor_attributes.columns]).reshape(1, -1)
scaled_user_vector = user_vector

similarities = cosine_similarity(scaled_flavor_attributes, scaled_user_vector)
df['Similarity'] = similarities

# Top recommendations
top_recs = df.sort_values('Similarity', ascending=False).drop_duplicates(['Beer Name (Full)']).head(10)
st.subheader("🍻 Top 10 Recommended Beers")
st.dataframe(top_recs[['Beer Name (Full)', 'Style', 'Similarity']])
