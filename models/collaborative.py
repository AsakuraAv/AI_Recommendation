import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input

train_data = pd.read_csv("data/train_data.csv")

# Parametry modelu
num_users = train_data["userId"].nunique()
num_movies = train_data["movieId"].nunique()

# Wejścia użytkownika i filmu
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

# Osadzenia (embeddings)
user_embedding = Embedding(num_users + 1, 16)(user_input)
movie_embedding = Embedding(num_movies + 1, 16)(movie_input)

# Spłaszczenie
user_flattened = Flatten()(user_embedding)
movie_flattened = Flatten()(movie_embedding)

# Połączenie i warstwa gęsta
concat = tf.keras.layers.Concatenate()([user_flattened, movie_flattened])
dense = Dense(128, activation="relu")(concat)
output = Dense(1, activation="linear")(dense)

# Tworzenie modelu
model = tf.keras.Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer="adam", loss="mse")

# Podgląd architektury
model.summary()