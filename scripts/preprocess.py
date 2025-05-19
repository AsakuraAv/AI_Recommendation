import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytanie danych
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")

# Oczyszczanie danych
movies.dropna(inplace=True)
ratings.dropna(inplace=True)
tags.dropna(inplace=True)

# Standaryzacja tagów i usunięcie duplikatów
tags["tag"] = tags["tag"].str.lower()
tags = tags.drop_duplicates(subset=["movieId", "tag"])  

# Grupowanie tagów dla każdego filmu
tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()

# Obliczenie średniej oceny każdego filmu
average_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
average_ratings.rename(columns={"rating": "average_rating"}, inplace=True)

# Obliczenie liczby ocen dla każdego filmu
rating_counts = ratings.groupby("movieId")["rating"].count().reset_index()
rating_counts.rename(columns={"rating": "num_ratings"}, inplace=True)

# Łączenie tagów z filmami
movies_tags = movies.merge(tags_grouped, on="movieId", how="left")
movies_tags["tag"] = movies_tags["tag"].fillna("")  

# Dodanie średnich ocen i liczby ocen do `movies_tags.csv`
movies_tags = movies_tags.merge(average_ratings, on="movieId", how="left")
movies_tags = movies_tags.merge(rating_counts, on="movieId", how="left")

# Zapis zaktualizowanych danych do pliku
movies_tags.to_csv("data/movies_tags.csv", index=False)

# Łączenie ocen użytkowników z danymi o filmach
merged_data = ratings.merge(movies_tags, on="movieId", how="left")
merged_data.to_csv("data/merged_data.csv", index=False)

# Podział na zbiór treningowy i testowy
train_data, test_data = train_test_split(merged_data[["userId", "movieId", "rating", "tag", "average_rating", "num_ratings"]], test_size=0.2, random_state=42)

train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)