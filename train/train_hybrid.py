import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input
import pickle  # Dodajemy zapis historii

movies = pd.read_csv("data/movies_tags.csv")
ratings = pd.read_csv("data/train_data.csv")

# Filtracja błędnych wartości movieId
num_users = ratings["userId"].nunique()
num_movies = ratings["movieId"].nunique()
ratings = ratings[ratings["movieId"] <= num_movies]

# Przygotowanie modelu
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

user_embedding = Embedding(num_users + 1, 16)(user_input)
movie_embedding = Embedding(num_movies + 1, 16)(movie_input)

user_flattened = Flatten()(user_embedding)
movie_flattened = Flatten()(movie_embedding)

concat = tf.keras.layers.Concatenate()([user_flattened, movie_flattened])
dense = Dense(128, activation="relu")(concat)
output = Dense(1, activation="linear")(dense)

model = tf.keras.Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer="adam", loss="mse")

# Trenowanie modelu i zapis historii
X_train = [ratings["userId"].values, ratings["movieId"].values]
y_train = ratings["rating"].values

history = model.fit(X_train, y_train, epochs=10, batch_size=32)

# Zapisujemy historię treningu
with open("train/history.pkl", "wb") as f:
    pickle.dump(history.history, f)