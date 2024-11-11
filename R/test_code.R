# Charger les fichiers de fonctions

source("R/LogisticRegressionMultinomial.R")
# Charger le package R6
library(R6)
library(glmnet)

# Charger le jeu de données Iris
data(iris)

# Préparer les données pour un modèle de régression logistique multinomiale
# Convertir 'Species' en facteur pour avoir des classes
iris$Species <- as.factor(iris$Species)

# Séparer les prédicteurs (X) et la variable cible (y)
X <- iris[, -5]  # Toutes les colonnes sauf Species
y <- iris$Species

#1. ################## Instancier le modèle avec les paramètres souhaités
model <- LogisticRegressionMultinomial$new(learning_rate = 0.01, num_iterations = 1000)

#2. ################# Ajuster le modèle sur le jeu de données
model$fit(X, y)

#3. ################## Afficher un résumé des paramètres ajustés du modèle
model$summary()

#4. ################## Prédire les probabilités d'appartenance aux classes pour les données d'entraînement
probabilities <- model$predict_proba(X)
print("Probabilités d'appartenance aux classes pour chaque observation :")
print(probabilities)

#5. ################## Prédire les classes pour les données d'entraînement
predictions <- model$predict(X)
print("Classes prédites pour chaque observation :")
print(predictions)

#6. ##################test de la fonction one_hot_encode
unique_classes <- unique(iris$Species)
one_hot_encoded_data <- model$one_hot_encode(iris$Species, unique_classes)
print("Données après encodage one-hot :")
print(head(one_hot_encoded_data))


#7. ################## Appeler la méthode print() pour afficher les informations du modèle
model$print()

#8. ################## Instancier le modèle avec une valeur par défaut pour le seuil de sélection (exemple : 0.1)
# Appliquer la sélection des variables sur le dataset iris
selected_data <- model$var_select(iris[, -5], iris$Species) 
print("Données après sélection des variables :")
print(head(selected_data))

# Comparer les prédictions aux vraies classes
accuracy <- mean(predictions == as.numeric(y))
cat("Précision du modèle sur les données d'entraînement :", accuracy, "\n")

