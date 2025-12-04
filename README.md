# Projet de Détection de Spams

## 1. Description du Projet

Ce projet a pour objectif de concevoir et développer un pipeline d'apprentissage automatique capable de classifier des e-mails en deux catégories : **Spam** (indésirable) ou **Ham** (légitime).

## 2. Architecture du Projet

Le projet est organisé selon l'arborescence suivante :

```text
spam_detection_project/
│
├── data/
│   ├── raw/             # Données brutes (Source Kaggle - Fichier CSV immuable)
│   └── processed/       # Données nettoyées et prêtes pour l'entraînement
│
├── models/
├── reports/
│   └── figures/         # Graphiques générés automatiquement (Matrices de confusion, Courbes ROC)
│
├── src/                 # Code source modulaire
│   ├── __init__.py
│   ├── eda.py           # Script d'Analyse Exploratoire des Données (EDA) - Indépendant
│   ├── preprocessing.py # Script de nettoyage et transformation des données
│   └── train.py         # Script d'entraînement, d'évaluation et de sauvegarde
│
├── main.py              # Script principal (Orchestrateur du pipeline de production)
├── requirements.txt     # Liste des dépendances logicielles
└── README.md            # Documentation du projet
```

## 3. Méthodologie et Pipeline Technique

Le traitement des données suit un pipeline séquentiel, implémenté dans le dossier `src/`.

### 3.1 Analyse Exploratoire des Données (EDA)

Avant tout traitement, un script dédié (src/eda.py) permet d'analyser le jeu de données :

- Visualisation de la distribution des classes (équilibre Spam/Ham).
- Détection des doublons et analyse des statistiques descriptives.
- Cette étape est purement analytique et ne fait pas partie de l'exécution automatique du main.py.

### 3.2 Prétraitement

Le nettoyage des données textuelles comprend les étapes suivantes :

- **Normalisation :** Conversion de l'ensemble du texte en minuscules.
- **Nettoyage :** Suppression de la ponctuation et des caractères spéciaux.
- **Filtrage :** Suppression des "stop-words" (mots vides de sens tels que "the", "is", "a") via la bibliothèque NLTK.

### 3.3 Vectorisation

La transformation du texte en données numériques est réalisée via la méthode **TF-IDF** (Term Frequency-Inverse Document Frequency) :

- Limitation aux **3000** fonctionnalités (mots) les plus pertinentes.
- Aucune normalisation de type `StandardScaler` n'est appliquée, car TF-IDF intègre déjà une normalisation et certains modèles (Naive Bayes) requièrent des entrées positives.

### 3.4 Modélisation

Cinq algorithmes distincts ont été implémentés et comparés :

1.  **Naive Bayes (MultinomialNB) :** Modèle probabiliste de référence pour la classification textuelle.
2.  **Arbre de Décision (Decision Tree) :** Modèle basé sur des règles, privilégié pour son interprétabilité.
3.  **Random Forest :** Modèle d'ensemble pour améliorer la robustesse et réduire le sur-apprentissage.
4.  **SVM (Support Vector Machine) :** Utilisation d'un noyau linéaire avec pondération des classes pour gérer le déséquilibre.
5.  **Isolation Forest :** Approche non-supervisée basée sur la détection d'anomalies.

## 4. Installation et Exécution

### Étape 1 : Installation des prérequis

Il faut avoir Python installé. Installez les dépendances nécessaires avec la commande suivante :

```bash
pip install -r requirements.txt
```

### Étape 2 : Analyse Exploratoire (Optionnel)

Pour générer les graphiques de distribution et les statistiques indépendemment du script principal, lancez :

```bash
python main.py
```

### Étape 3 : Lancement du projet

Le lancement du projet se fait avec :

```bash
python main.py
```

## 5. Résultats et Livrables

À l'issue de l'exécution du script `main.py`, les livrables suivants sont générés automatiquement :

- **Console :** Un tableau récapitulatif des scores d'Accuracy pour chaque modèle s'affiche dans le terminal.
- **Graphiques (`reports/figures/`) :**
  - Matrices de confusion pour chaque modèle.
  - Courbes ROC avec calcul de l'AUC pour évaluer la discrimination.
  - Graphique de distribution des classes à partir de l'EDA.

## 6. Source des Données

Le jeu de données utilisé est le **Spam Mails Dataset**, provenant de la plateforme Kaggle. Il contient un ensemble d'e-mails étiquetés manuellement comme "spam" ou "ham".
