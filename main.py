import os

print("Uruchamiam preprocessing danych...")
os.system("python models/processing.py")

print("Trenuję model i testuję rekomendacje...")
os.system("python train/train_hybrid.py")

print("Generuję wizualizację błędu treningowego...")
os.system("python plots/train_loss_plot.py")