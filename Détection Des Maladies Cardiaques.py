# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# Chargement du dataset à partir du fichier CSV
file_path = 'D:\MST\S3\Data Mining\python\heart_disease_health_indicators_BRFSS2015.csv'
data = pd.read_csv(file_path)
# Obtenir la forme du tableau
print(data.shape)

# Affichage des premières lignes du dataset pour vérification
print(data.head())

# afficher les 5 derniers lignes
print(data.tail())


#################### Nettoyage et Transformation des Données#######################################
# 1. Vérification des valeurs manquantes
missing_values = data.isnull().sum()
print("\nValeurs manquantes par colonne :")
print(missing_values)

# 2. Suppression des lignes avec des valeurs manquantes (si nécessaire)
data = data.dropna()
data.isnull().sum()

# 3. Transformation des variables catégorielles en variables indicatrices (si nécessaire)
# Exemple : Encodage one-hot des colonnes catégorielles
data_encoded = pd.get_dummies(data, columns=['Sex', 'Education'])


# 4. Normalisation ou standardisation des données
# Exemple : Min-max scaling des colonnes numériques
scaler = MinMaxScaler()
columns_to_scale = ['Age', 'Income', 'BMI']
data_encoded[columns_to_scale] = scaler.fit_transform(data_encoded[columns_to_scale])

# Affichage des premières lignes du dataset après nettoyage et transformation
print("\nAprès le nettoyage et la transformation :")
print(data_encoded.head())

print(data['HeartDiseaseorAttack'].value_counts())

###############################Exploration des Données (EDA)##########################
# 1. Statistiques descriptives
descriptive_stats = data.describe()
print("\nStatistiques descriptives :")
print(descriptive_stats)

# 2. Visualisation des corrélations entre les variables
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.show()

# 3. Visualisation des distributions des variables importantes
# Variables pertinentes pour la détection des maladies cardiaques
variables_of_interest = ['Age', 'HighBP', 'HighChol', 'BMI', 'Smoker', 'Diabetes', 'PhysActivity', 'HeartDiseaseorAttack']

# Création d'une sous-section du dataset avec les variables pertinentes
selected_data = data[variables_of_interest]

# Affichage des premières lignes du dataset pour vérification
print("Avant l'exploration des données :")
print(selected_data.head())

# Visualisation des distributions des variables pertinentes
plt.figure(figsize=(15, 10))
selected_data.hist(bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.suptitle('Distributions des Variables pour la Détection des Maladies Cardiaques \n', x=0.5, y=0.92, fontsize=16)
# Ajouter un retour à la ligne
plt.tight_layout()
plt.show()


###############################Appliquer les techniques de fouille de données###########################

# Création d'une sous-section du dataset avec les variables pertinentes
selected_data = data[variables_of_interest]

# Division des données en ensemble d'entraînement et ensemble de test (80% / 20% ici)
train_data, test_data = train_test_split(selected_data, test_size=0.2, random_state=42)

# Affichage de la taille des ensembles d'entraînement et de test
print("Taille de l'ensemble d'entraînement :", len(train_data))
print("Taille de l'ensemble de test :", len(test_data))

# Séparation des variables indépendantes (X) et de la variable cible (y)
X_train = train_data.drop('HeartDiseaseorAttack',axis=1)
y_train = train_data['HeartDiseaseorAttack']
X_test = test_data.drop('HeartDiseaseorAttack', axis=1)
y_test = test_data['HeartDiseaseorAttack']


# Entraînement du modèle (Random Forest)
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred_rf = model_rf.predict(X_test)

# Affichage des résultats
print("Random Forest Classifier:")
print("\nRapport de classification :\n", classification_report(y_test, y_pred_rf))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_rf))
print("\nAUC-ROC Score:", roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1]))

####################################  Interprétation du Modèle et Établissement des Conclusions ######################
# Afficher l'importance des caractéristiques
feature_importances = pd.Series(model_rf.feature_importances_, index=X_train.columns)
print("Feature Importances:\n", feature_importances)

# Visualiser l'importance des caractéristiques 

# Trier les caractéristiques par importance
sorted_idx = feature_importances.argsort()
sorted_feature_importances = feature_importances[sorted_idx]

# Créer un graphique à barres horizontales pour visualiser l'importance des caractéristiques
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_feature_importances)), sorted_feature_importances)
plt.yticks(range(len(sorted_feature_importances)), X_train.columns[sorted_idx])
plt.xlabel('Importance')
plt.title('Importance des Caractéristiques - Random Forest Classifier')
plt.show()


