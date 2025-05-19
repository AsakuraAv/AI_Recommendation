import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input

train_data = pd.read_csv("data/train_data.csv")

print("Dane treningowe poprawnie wczytane!")

# Parametry modelu
num_users = train_data["userId"].nunique()
num_movies = train_data["movieId"].nunique()

# Sprawdzenie poprawności wartości w `train_data.csv`
print("Maksymalne wartości w danych treningowych:")
print("Najwyższy userId:", train_data["userId"].max())
print("Najwyższy movieId:", train_data["movieId"].max())
print("Liczba unikalnych użytkowników:", num_users)
print("Liczba unikalnych filmów:", num_movies)

# Filtracja błędnych wartości (usuwamy ID większe niż liczba osadzeń)
train_data = train_data[
    (train_data["userId"] <= num_users) & (train_data["movieId"] <= num_movies)
]

print("Dane przefiltrowane - usunięto błędne wartości movieId!")

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

# Przygotowanie danych treningowych
X_train = [train_data["userId"].values, train_data["movieId"].values]
y_train = train_data["rating"].values

# Trenowanie modelu
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Sprawdzenie przewidywań
sample_users = np.random.randint(1, num_users + 1, 5)
sample_movies = np.random.randint(1, num_movies + 1, 5)
predictions = model.predict([sample_users, sample_movies])

print("Przykładowe przewidywania ocen dla użytkowników:")
for user, movie, pred in zip(sample_users, sample_movies, predictions):
    print(f"Użytkownik {user}, Film {movie} → Przewidywana ocena: {pred[0]:.2f}")