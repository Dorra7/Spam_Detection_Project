import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import os

# Configuration
RAW_DATA_PATH = 'data/raw/spam_ham_dataset.csv'
PROCESSED_DATA_PATH = 'data/processed/spam_cleaned.csv'

def download_nltk_resources():
    #Télécharger les stop-words de NLTK.
    nltk.download('stopwords', quiet=True)

def clean_text(text):
    
    # S'assurer que l'entrée est une chaîne de caractères
    if not isinstance(text, str):
        return ""
    
    # Minuscules
    text = text.lower()
    
    # Suppression de la ponctuation
    text = "".join([char for char in text if char not in string.punctuation])
    
    # Stop-words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return " ".join(filtered_words)

def pretraitement():
    print("Prétraitement...")
    
    # Chargement
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERREUR: Le fichier {RAW_DATA_PATH} est introuvable.")
        return
        
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Données chargées : {df.shape}")

    # Préparation NLTK
    download_nltk_resources()

    # Application du nettoyage
    print("Nettoyage du texte en cours...")
    df['text_cleaned'] = df['text'].apply(clean_text)

    # Sélection des colonnes utiles ('label_num' et'text_cleaned')
    df_final = df[['label_num', 'text_cleaned']].dropna()

    # Sauvegarde dans data/processed
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Fichier d'apprentissage sauvegardé")
    print(df_final.head())

if __name__ == "__main__":
    pretraitement()