# Projet de Détection de Spams

## 1. Description du Projet

Ce projet a pour objectif de concevoir et développer une chaîne de traitement (pipeline) d'apprentissage automatique capable de classifier des e-mails en deux catégories : **Spam** (indésirable) ou **Ham** (légitime).

La problématique centrale repose sur l'analyse de texte (Text Mining). L'enjeu critique défini pour ce projet est de maximiser la détection des spams tout en **minimisant impérativement les Faux Positifs** (classer un e-mail important comme spam), afin de garantir la fiabilité du système pour l'utilisateur final.

Le projet respecte l'architecture standard **Cookiecutter Data Science** pour assurer la reproductibilité, la modularité et la lisibilité du code.

## 2. Architecture du Projet

Le projet est organisé selon l'arborescence suivante :

````text
spam_detection_project/
│
├── data/
│   ├── raw/             # Données brutes (Source Kaggle - Fichier CSV immuable)
│   └── processed/       # Données nettoyées et prêtes pour l'entraînement
│
├── models/              # Modèles entraînés (.pkl) et vectorizer TF-IDF sauvegardés
│
├── reports/
│   └── figures/         # Graphiques générés automatiquement (Matrices de confusion, Courbes ROC)
│
├── src/                 # Code source modulaire
│   ├── __init__.py
│   ├── preprocessing.py # Script de nettoyage et transformation des données
│   └── train.py         # Script d'entraînement, d'évaluation et de sauvegarde
│
├── main.py              # Script principal (Orchestrateur du projet)
├── requirements.txt     # Liste des dépendances logicielles
└── README.md            # Documentation du projet

## 3. Méthodologie et Pipeline Technique

Le traitement des données suit un pipeline séquentiel strict, implémenté dans le dossier `src/`.

### 3.1 Prétraitement (Preprocessing)
Le nettoyage des données textuelles comprend les étapes suivantes :
* **Normalisation :** Conversion de l'ensemble du texte en minuscules.
* **Nettoyage :** Suppression de la ponctuation et des caractères spéciaux.
* **Filtrage :** Suppression des "stop-words" (mots vides de sens tels que "the", "is", "a") via la bibliothèque NLTK.
* **Dédoublonnage :** Suppression des doublons au sein des messages pour éviter la redondance.

### 3.2 Vectorisation
La transformation du texte en données numériques est réalisée via la méthode **TF-IDF** (Term Frequency-Inverse Document Frequency) :
* Limitation aux **3000** fonctionnalités (mots) les plus pertinentes.
* Aucune normalisation de type `StandardScaler` n'est appliquée, car TF-IDF intègre déjà une normalisation L2 et certains modèles (Naive Bayes) requièrent des entrées positives.

### 3.3 Modélisation
Cinq algorithmes distincts ont été implémentés et comparés :

1.  **Naive Bayes (MultinomialNB) :** Modèle probabiliste de référence pour la classification textuelle.
2.  **Arbre de Décision (Decision Tree) :** Modèle basé sur des règles, privilégié pour son interprétabilité.
3.  **Random Forest :** Modèle d'ensemble (bagging) pour améliorer la robustesse et réduire le sur-apprentissage.
4.  **SVM (Support Vector Machine) :** Utilisation d'un noyau linéaire (`kernel='linear'`) avec pondération des classes (`class_weight='balanced'`) pour gérer le déséquilibre.
5.  **Isolation Forest :** Approche non-supervisée basée sur la détection d'anomalies.

## 4. Installation et Exécution

Ce projet a été conçu pour être exécuté via un point d'entrée unique.

### Étape 1 : Installation des prérequis
Assurez-vous d'avoir Python installé. Installez les dépendances nécessaires avec la commande suivante :

```bash
pip install -r requirements.txt


### Étape 2 : Lancement du projet
Un script orchestrateur (`main.py`) exécute séquentiellement le nettoyage des données puis l'entraînement des modèles. Lancez simplement :

```bash
python main.py


## 5. Résultats et Livrables

À l'issue de l'exécution du script `main.py`, les livrables suivants sont générés automatiquement :

* **Console :** Un tableau récapitulatif des scores d'Accuracy pour chaque modèle s'affiche dans le terminal.
* **Graphiques (`reports/figures/`) :**
    * Matrices de confusion pour chaque modèle (format PNG).
    * Courbes ROC avec calcul de l'AUC pour évaluer la discrimination (format PNG).
* **Modèles (`models/`) :**
    * Les modèles entraînés sont sérialisés au format `.pkl` (ex: `naive_bayes.pkl`).
    * Le vectorizer TF-IDF est également sauvegardé pour permettre la prédiction sur de nouveaux messages.

## 6. Source des Données

Le jeu de données utilisé est le **Spam Mails Dataset**, provenant de la plateforme Kaggle. Il contient un ensemble d'e-mails étiquetés manuellement comme "spam" ou "ham".
````
