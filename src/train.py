import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest

# Configuration des chemins
INPUT_FILE = 'data/processed/spam_cleaned.csv'
FIGURES_DIR = 'reports/figures/'  

def train_and_evaluate():
    # Préparation
    print("Chargement et Vectorisation")
    if not os.path.exists(INPUT_FILE):
        print(f"Erreur: {INPUT_FILE} introuvable.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    df['text_cleaned'] = df['text_cleaned'].fillna('')
    
    # Création du dossier figures
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Vectorisation
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['text_cleaned']).toarray()
    y = df['label_num'].values
    
    
    # Split en traine et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Définition des Modèles
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=20),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42),
        "Isolation Forest": IsolationForest(contamination=0.2, random_state=42, n_jobs=-1)
    }

    results = []

    # Boucle d'entraînement et d'évaluation 
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Traitement du modèle : {name}")
        
        # Entraînement et prédiction
        if name == "Isolation Forest":
            model.fit(X_train)
            y_pred_raw = model.predict(X_test)
            # Conversion des prédictions (-1 = Spam, 1 = Ham)
            y_pred = [1 if x == -1 else 0 for x in y_pred_raw]
            # On inverse le signe car ce modèle note les Spams négativement (score élevé donc spam)
            y_scores = -model.decision_function(X_test)
            
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Scores pour la ROC
            y_scores = model.predict_proba(X_test)[:, 1] # Probabilité d'être dans la classe 1 (Spam)

        # Métriques
        acc = accuracy_score(y_test, y_pred)
        results.append({'Model': name, 'Accuracy': acc})
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
       

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matrice de Confusion : {name}')
        plt.ylabel('Vrai Label (0=Ham, 1=Spam)')
        plt.xlabel('Label Prédit')
        
        cm_path = os.path.join(FIGURES_DIR, f'confusion_matrix_{name.replace(" ", "_")}.png')
        plt.savefig(cm_path)
        plt.close() 
        print(f"Matrice de confusion sauvegardée")

        # Courbe ROC & AUC 
        try:    
            tfp, tvp, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(tfp, tvp)
            
            plt.figure(figsize=(8, 6))
            plt.plot(tfp, tvp, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taux Faux Positifs')
            plt.ylabel('Taux Vrais Positifs')
            plt.title(f'Courbe ROC : {name}')
            plt.legend(loc="lower right")
            
            roc_path = os.path.join(FIGURES_DIR, f'roc_curve_{name.replace(" ", "_")}.png')
            plt.savefig(roc_path)
            plt.close()
            print(f"-> Courbe ROC sauvegardée : {roc_path}")
            
        except Exception as e:
            print(f"Impossible de générer la ROC pour {name}: {e}")

    # Comparaison finale des modèles
    print(f"\n{'='*40}")
    print("Classement des Modèles final :")
    print(pd.DataFrame(results).sort_values(by='Accuracy', ascending=False))

if __name__ == "__main__":
    train_and_evaluate()