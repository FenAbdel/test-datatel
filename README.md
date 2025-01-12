# DataTel Analytics : Test technique

## Context

Voici ma réalisation pour le test technique de DataTel. Le projet consiste en une analyse de données incluant des étapes de traitement, de création de nouvelles features et de modélisation, dans le but de faire des prévisions sur un ensemble de données. La structure du projet ainsi que les étapes nécessaires pour reproduire l'analyse sont présentées ci-dessous.

```bash
├── README.md # Ce fichier 
├── data/ 
│ ├── final/	 # Données finales 
│ ├── raw/	 # Données brutes 
│ └── processed/	 # Données transformées 
├── notebooks/ 
│ ├── 01_exploration.ipynb 	# Analyse exploratoire des données 
│ └── 02_modeling.ipynb 	# Modélisation et prévisions 
├── src/ 
│ ├── __init__.py 	# Initialisation du module 
│ ├── data_processing.py 	# Fonctions de traitement des données 
│ ├── feature_engineering.py 		# Création des features 
│ └── models.py #	 Code des modèles de machine learning 
└── presentation/ 
└── results.pdf 	# Présentation des résultats

```


## Prérequis

Avant de commencer, assurez-vous d'avoir les prérequis suivants :

- Python 3.x
- pip (gestionnaire de paquets)


## Étapes pour reproduire l'analyse

### 1. Créer un environnement virtuel

Créez un environnement virtuel pour isoler les dépendances du projet.

```bash
python -m venv venv
```

### 2. Installer les dépendances

Activez l'environnement virtuel et installez les dépendances nécessaires à l'aide de `requirements.txt`.

```
`# Activer l'environnement virtuel
source venv/bin/activate  # Sous Linux/macOS
venv\Scripts\activate     # Sous Windows

# Installer les dépendances
pip install -r requirements.txt


```

### 3. Prétraitement des données

Exécutez le script `data_processing.py` pour nettoyer et préparer les données brutes.

```
python src/data_processing.py
```

### 4. Création des features

Exécutez le script `feature_engineering.py` pour créer les features nécessaires pour la modélisation.

```
python src/feature_engineering.py
```

### 5. Entraînement des modèles

Exécutez le script `models.py` pour entraîner et évaluer les modèles de machine learning.

```
python src/models.py
```

### 6. Exécution des notebooks

Accédez aux notebooks Jupyter pour effectuer l'analyse exploratoire et la modélisation, et pour visualiser les résultats.

1. Ouvrez le notebook `01_exploration.ipynb` pour l'analyse exploratoire des données.
2. Ouvrez le notebook `02_modeling.ipynb` pour effectuer les prévisions à l'aide des modèles.


## Remarques

* Le répertoire `data/raw/` contient les données brutes téléchargées.
* Le répertoire `data/processed/` contient les données après transformation et nettoyage.
* Le répertoire `data/final/` contient les résultats finaux de l'analyse.
