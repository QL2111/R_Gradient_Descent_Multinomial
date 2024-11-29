# https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data Variable à predire: Industry
# On test avec Approved car l'accuracy est très faible pour Industry -> Même avec approved, l'accuracy est faible (0.16).. Il y a un problème avec le modèle

# Charger les bibliothèques nécessaires
# nolint start
library(R6)

# Charger les fichiers de fonctions
source("R/DataPreparer.R")
source("R/factor_analysis_mixed.R")
source("R/LogisticRegressionMultinomial.R")

print("Credit Card Approval Prediction Example")

# Charger le jeu de données depuis un fichier local après téléchargement de Kaggle
data_path <- "data/credit_card.csv"  # Remplacez par le chemin de votre fichier
data <- read.csv(data_path)

# S'assurer que la variable cible est un facteur
data$Approved <- as.factor(data$Approved)

# Diviser et préparer les données en ensembles d'entraînement et de test
set.seed(42)  # Pour la reproductibilité

data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
prepared_data <- data_prep$prepare_data(data, "Approved", 0.7, stratify = TRUE)

# Accéder aux données préparées
X_train <- prepared_data$X_train
X_test <- prepared_data$X_test
y_train <- prepared_data$y_train
y_test <- prepared_data$y_test

# Afficher les proportions des classes dans les ensembles d'entraînement et de test
cat("Proportions des classes dans l'ensemble d'entraînement :\n")
print(table(y_train) / length(y_train))
cat("Proportions des classes dans l'ensemble de test :\n")
print(table(y_test) / length(y_test))

# Convertir les données préparées en matrices
X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)

# Convertir la variable cible en valeurs numériques
y_train_numeric <- as.numeric(y_train)
y_test_numeric <- as.numeric(y_test)

# Initialiser et ajuster le modèle sur l'ensemble d'entraînement
model <- LogisticRegressionMultinomial$new(learning_rate = 0.1, num_iterations = 1000, loss="logistique", optimizer="adam", use_early_stopping=TRUE)
model$fit(X_train_matrix, y_train_numeric)

# Prédire sur l'ensemble de test
predictions <- model$predict(X_test_matrix)

# Afficher les prédictions
# print(predictions)

# Calculer et afficher l'accuracy
accuracy <- sum(predictions == y_test_numeric) / length(y_test_numeric)
cat("Accuracy:", accuracy, "\n")

# Matrice de confusion pour évaluer les performances
model$summary()

model$print(X_test_matrix, y_test_numeric)

# print(confusion_matrix)

# Importance des variables
# model$var_importance()

# nolint end
