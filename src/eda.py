import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration des chemins
RAW_DATA_PATH = 'data/raw/spam_ham_dataset.csv'
FIGURES_DIR = 'reports/figures/'

def explore_data():
    print("ANALYSE EXPLORATOIRE (EDA)")

    # Chargement
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Erreur : Le fichier {RAW_DATA_PATH} est introuvable.")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    
    # Création du dossier pour sauvegarder les images
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Aperçu global
    print(f"\nDimensions du dataset : ")
    print(f"Lignes : {df.shape[0]}")
    print(f"Colonnes : {df.shape[1]}")
    
    print(f"\nAperçu des colonnes : ")
    print(df.info())

    # Analyse des doublons (Important pour les spams)
    duplicates = df.duplicated().sum()
    print(f"\nAnalyse des Doublons")
    print(f"Nombre de lignes dupliquées : {duplicates}")
    if duplicates > 0:
        print("Note : Le preprocessing se chargera de les traiter.")

    # Analyse de la distribution des classes (Objectif du projet)
    print(f"\nDistribution des classes (Spam vs Ham)")
    # On utilise 'label_num' où souvent 1=Spam, 0=Ham
    counts = df['label_num'].value_counts()
    percentages = df['label_num'].value_counts(normalize=True) * 100
    
    print(counts)
    print("\nEn pourcentage :")
    print(percentages)

    # Génération du graphique pour le rapport
    print(f"\nGénération du graphique")
    plt.figure(figsize=(6, 5))
    sns.countplot(x='label_num', data=df, palette='viridis')
    plt.title('Distribution des classes (0=Ham, 1=Spam)')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'emails')
    
    # Sauvegarde
    output_path = os.path.join(FIGURES_DIR, 'class_distribution.png')
    plt.savefig(output_path)
    print(f"Graphique sauvegardé dans : {output_path}")
    plt.close()

if __name__ == "__main__":
    explore_data()