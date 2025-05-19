import matplotlib.pyplot as plt
import pickle

# Wczytujemy historię
with open("train/history.pkl", "rb") as f:
    history = pickle.load(f)

# Wizualizacja błędu treningowego
plt.plot(history['loss'])
plt.xlabel('Epoki')
plt.ylabel('Błąd MSE')
plt.title('Spadek błędu podczas treningu')
plt.show()