import pandas as pd

# Wczytaj plik
movies_tags = pd.read_csv("data/movies_tags.csv")

# Wyświetl pierwsze 5 wierszy
print(movies_tags.head())