# Charger les fichiers de fonctions
source("D:/M2 SISE/Programmation R/Projet R/R_Gradient_Descent_Multinomial/R/LogisticRegressionMultinomial.R")
# Charger le package R6
library(R6)
library(glmnet)
library(nnet)
library(caret)

# Charger un fichier CSV dans R
data <- read.csv("D:/M2 SISE/Programmation R/Projet R/dataset test/user_behavior_dataset.csv", header = TRUE, sep = ",")
data
colnames(data)

# Extraire la variable cible 
y <- data$Device.Model
y

# Encodage one-hot des variables catégorielles
X <-data[, -c(1, 2)]
X <- X[, sapply(X, is.numeric)]
colnames(X)

# Vérifier les valeurs m#anquantes
any(is.na(X))       # Pour les valeurs manquantes

X <- scale(X)
y <- as.factor(y)

#1. ################## Instancier le modèle avec les paramètres souhaités
model <- LogisticRegressionMultinomial$new(learning_rate = 0.001, num_iterations = 1000)

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
unique_classes <- unique(y)
one_hot_encoded_data <- model$one_hot_encode(y, unique_classes)
print("Données après encodage one-hot :")
print(head(one_hot_encoded_data))

#7. ################## Appeler la méthode print() pour afficher les informations du modèle
model$print()

#8. ################## Instancier le modèle avec une valeur par défaut pour le seuil de sélection (exemple : 0.1)
# Appliquer la sélection des variables sur le dataset
selected_data <- model$var_select(X, y)
print("Données après sélection des variables :")
print(head(selected_data))


# Comparer les prédictions aux vraies classes
accuracy <- mean(predictions == as.numeric(y))
cat("L'accuracy du modèle sur les données d'entraînement :", accuracy, "\n")



