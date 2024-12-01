# nolint start
# https://www.kaggle.com/datasets/lainguyn123/student-performance-factors Variable à predire: Access_to_Resources

# Charger les bibliothèques nécessaires
library(R6)

# Charger les fichiers de fonctions
source("R/DataPreparer.R")
source("R/LogisticRegressionMultinomial.R")

# Charger le jeu de données après téléchargement de Kaggle
data_path <- "data/Iris.csv"  # Remplacez par le chemin de votre fichier
data <- read.csv(data_path)

# S'assurer que la variable cible est un facteur
data$Species <- as.factor(data$Species)

# Diviser les données en ensembles d'entraînement et de test

prepared_data <- data_prep$prepare_data(data, "Species", 0.7, stratify = TRUE, remove_outliers = FALSE, outlier_seuil = 0.10)
X_train <- prepared_data$X_train
X_test <- prepared_data$X_test
y_train <- prepared_data$y_train
y_test <- prepared_data$y_test

X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)

# Convertir la variable cible en valeurs numériques
y_train_numeric <- as.numeric(y_train)
y_test_numeric <- as.numeric(y_test)

# Initialiser et ajuster le modèle sur l'ensemble d'entraînement
model <- LogisticRegressionMultinomial$new(learning_rate = 0.1, num_iterations = 1000, loss="logistique", optimizer="sgd", use_early_stopping=TRUE, regularization = "ridge")
model$fit(X_train_matrix, y_train_numeric)

# Prédire sur l'ensemble de test
predictions <- model$predict(X_test_matrix)

# Afficher les prédictions
# print(predictions)

# Calculer et afficher l'accuracy
# accuracy <- sum(predictions == y_test_numeric) / length(y_test_numeric)
# cat("Accuracy:", accuracy, "\n")
model$summary()
model$plot_loss()
model$print(X_test_matrix, y_test_numeric)
print("Variable Importance:")
model$var_importance()

# nolint end
