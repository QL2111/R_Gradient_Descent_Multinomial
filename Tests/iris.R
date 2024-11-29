# nolint start
# Charger les fichiers de fonctions
source("R/DataPreparer.R")
source("R/factor_analysis_mixed.R")
source("R/LogisticRegressionMultinomial.R")

# Charger les données Iris

data(iris)
iris$Species <- as.factor(iris$Species)

# Diviser les données en train/test avant le traitement
set.seed(42)  # Pour la reproductibilité


data_prep <- DataPreparer$new(use_factor_analysis = FALSE) # On utilise factor analysis pour les variables qualitatives
prepared_data <- data_prep$prepare_data(iris, "Species", 0.7, FALSE) # Jeu de données, variable cible, proportion d'entraînement, stratification
# print(prepared_data)
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

# # Initialiser et ajuster le modèle sur l'ensemble d'entraînement
# model <- LogisticRegressionMultinomial$new(learning_rate = 0.01, num_iterations = 1000)
# model$fit(X_train_matrix, y_train_numeric)

# # Prédire sur l'ensemble de test
# predictions <- model$predict(X_test_matrix)

# # Afficher les prédictions
# # print(predictions)

# # Calculer et afficher l'accuracy
# accuracy <- sum(predictions == y_test_numeric) / length(y_test_numeric)
# cat("Accuracy:", accuracy, "\n")

# # Matrice de confusion pour évaluer les performances
# confusion_matrix <- table(Predicted = predictions, Actual = y_test_numeric)
# # print(confusion_matrix)
# model$print(X_test_matrix, y_test_numeric)
# # Importance des variables
# model$var_importance()

# nolint end