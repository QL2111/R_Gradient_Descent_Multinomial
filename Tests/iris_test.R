# nolint start
# https://www.kaggle.com/datasets/lainguyn123/student-performance-factors Variable à predire: Access_to_Resources

# Charger les bibliothèques nécessaires
library(R6)

# Charger les fichiers de fonctions
source("DataPreparer.R")
source("LogisticRegressionMultinomial.R")

# Charger le jeu de données après téléchargement de Kaggle
data_path <- "Iris.csv"  # Remplacez par le chemin de votre fichier
data <- read.csv(data_path)

# S'assurer que la variable cible est un facteur
data$Species <- as.factor(data$Species)

# Diviser les données en ensembles d'entraînement et de test
set.seed(42)  # Pour la reproductibilité
train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))  # 70% pour l'entraînement
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Séparer les caractéristiques et la variable cible
X_train <- train_data[, -which(names(train_data) == "Species")]
y_train <- train_data$Species

X_test <- test_data[, -which(names(test_data) == "Species")]
y_test <- test_data$Species

# Préparer les prédicteurs sans inclure la variable cible
data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
prepared_X_train <- data_prep$prepare_data(X_train)
prepared_X_test <- data_prep$prepare_data(X_test)

# Convertir les données préparées en matrices
X_train_matrix <- as.matrix(prepared_X_train)
X_test_matrix <- as.matrix(prepared_X_test)

# Convertir la variable cible en valeurs numériques
y_train_numeric <- as.numeric(y_train)
y_test_numeric <- as.numeric(y_test)

# Initialiser et ajuster le modèle sur l'ensemble d'entraînement
model <- LogisticRegressionMultinomial$new(learning_rate = 0.1, num_iterations = 1000, loss="logistique", optimizer="adam", use_early_stopping=TRUE, regularization = "ridge", lambda1 = 0.0, lambda2 = 0.01)
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
