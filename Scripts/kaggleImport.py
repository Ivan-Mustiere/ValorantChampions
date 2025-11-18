import kagglehub
import shutil
import os

# Téléchargement du dataset depuis Kaggle 2024 et 2025
print("Téléchargement du dataset...")
path2024 = kagglehub.dataset_download("piyush86kumar/valorant-champions-2024")
path2025 = kagglehub.dataset_download("piyush86kumar/valorant-champions-tour-2025-paris")

print(f"Dataset téléchargé dans : {path2024}")
print(f"Dataset téléchargé dans : {path2025}")

# Définir le dossier de destination dans ton espace de travail
destination2024 = os.path.expanduser("~/ValorantChampions/data/2024")
destination2025 = os.path.expanduser("~/ValorantChampions/data/2025")

# Copier le dossier (dirs_exist_ok=True permet d’écraser si le dossier existe déjà)
print(f"Copie des fichiers vers : {destination2024}")
shutil.copytree(path2024, destination2024, dirs_exist_ok=True)
print(f"Copie des fichiers vers : {destination2025}")
shutil.copytree(path2025, destination2025, dirs_exist_ok=True)
print("✅ Copie terminée.")
print(f"Les fichiers sont maintenant disponibles ici : {destination2024}")
print(f"Les fichiers sont maintenant disponibles ici : {destination2025}")
