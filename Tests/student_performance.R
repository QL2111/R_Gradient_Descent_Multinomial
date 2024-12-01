# nolint start
# https://www.kaggle.com/datasets/lainguyn123/student-performance-factors Variable à predire: Access_to_Resources

# Charger les bibliothèques nécessaires
library(R6)

# Charger les fichiers de fonctions
source("R/DataPreparer.R")
source("R/LogisticRegressionMultinomial.R")

data_path <- "data/StudentPerformanceFactors.csv"  # Remplacez par le chemin de votre fichier
data <- read.csv(data_path)

# S'assurer que la variable cible est un facteur
data$Access_to_Resources <- as.factor(data$Access_to_Resources)

# Diviser les données en ensembles d'entraînement et de test
set.seed(42)  # Pour la reproductibilité

data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
prepared_data <- data_prep$prepare_data(data, "Access_to_Resources", 0.7, stratify = TRUE, remove_outliers = TRUE, outlier_seuil = 0.10)
# Check if the proportions are equals
# print(table(prepared_data$y_train) / length(prepared_data$y_train))
# print(table(prepared_data$y_test) / length(prepared_data$y_test))

# Accéder aux données préparées
X_train <- prepared_data$X_train
X_test <- prepared_data$X_test
y_train <- prepared_data$y_train
y_test <- prepared_data$y_test

# Check AFDM (avec one hot encoder 32 dimensions) # 27 dimensions sans one hot encoder, encore réduire ?
# print(ncol(X_train))

# Afficher les proportions des classes dans les ensembles d'entraînement et de test
# cat("Proportions des classes dans l'ensemble d'entraînement :\n")
# print(table(y_train) / length(y_train))
# cat("Proportions des classes dans l'ensemble de test :\n")
# print(table(y_test) / length(y_test))

# Convertir les données préparées en matrices
X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)

# Convertir la variable cible en valeurs numériques
y_train_numeric <- as.numeric(y_train)
y_test_numeric <- as.numeric(y_test)

# Initialiser et ajuster le modèle sur l'ensemble d'entraînement
# initialize = function(learning_rate = 0.01, num_iterations = 1000, loss = "logistique", 
#     optimizer = "adam", beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, patience = 20, 
#     use_early_stopping = TRUE, regularization = "none", batch_size = 32) 
model <- LogisticRegressionMultinomial$new(learning_rate = 0.1, num_iterations = 500, loss="logistique", optimizer="adam",batch_size=32, use_early_stopping=TRUE, regularization = "elasticnet")
# lasso F1 = 0.99
# ridge F1 = 0.99
# elasticnet F1 = 0.99
# FALSE = 0.99

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