import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/movies_tags.csv")
ratings = pd.read_csv("data/ratings_clean.csv")

# Filtracja filmów - usuwanie bez ocen*
rated_movies = ratings["movieId"].unique()
movies = movies[movies["movieId"].isin(rated_movies)]

# Filtracja użytkowników - usuwanie z małą liczbą ocen
user_counts = ratings["userId"].value_counts()
filtered_users = user_counts[user_counts >= 5].index  # Usuwamy użytkowników z < 5 ocen
ratings = ratings[ratings["userId"].isin(filtered_users)]

# Normalizacja ocen do zakresu [0,1]
ratings["rating"] = ratings["rating"] / ratings["rating"].max()

# Łączenie gatunków i tagów w jedną cechę
movies["combined_features"] = movies["genres"].fillna("") + " " + movies["tag"].fillna("")

# Przetwarzanie danych za pomocą TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

# Obliczenie podobieństwa Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Funkcja rekomendacji na podstawie tagów i gatunków
def recommend_movies(movie_title, movies, cosine_sim):
    if movie_title not in movies["title"].values:
        return f"Film '{movie_title}' nie został znaleziony w bazie danych!"

    idx = movies[movies["title"] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # 5 najbardziej podobnych filmów
    movie_indices = [i[0] for i in sim_scores]

    return movies["title"].iloc[movie_indices].tolist()

# Test rekomendacji
print("Filmy podobne do 'Toy Story (1995)':")
print(*recommend_movies("Toy Story (1995)", movies, cosine_sim), sep="\n")