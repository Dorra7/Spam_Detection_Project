import sys
import os
from src.preprocessing import pretraitement as run_preprocessing
from src.train import train_and_evaluate as run_training

def main():
    print("DÉMARRAGE DU  SPAM DETECTION")
    print("==================================================")

    # Nettoyage
    print("\nLancement du prétraitement des données...")
    try:
        run_preprocessing()
        print("Nettoyage terminé")
    except Exception as e:
        print(f"Erreur critique lors du nettoyage : {e}")
        sys.exit(1)

    # Entraînement
    print("\nLancement de l'entraînement et de l'évaluation...")
    try:
        run_training()
        print("Entraînement terminé")
    except Exception as e:
        print(f"Erreur lors de l'entraînement : {e}")

    print("==================================================")
    print("Fin")

if __name__ == "__main__":
    main()