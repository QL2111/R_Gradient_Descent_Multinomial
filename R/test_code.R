# Charger les fichiers de fonctions
source("R/DataPreparer.R")
source("R/factor_analysis_mixed.R")
source("R/LogisticRegressionMultinomial.R")

# Charger les données Iris
data(iris)
iris$Species <- as.factor(iris$Species)

# Diviser les données en train/test
set.seed(42) # Pour la reproductibilité
train_indices <- sample(1:nrow(iris), size = 0.7 * nrow(iris)) # 70% pour l'entraînement
train_data <- iris[train_indices, ]
test_data <- iris[-train_indices, ]

# Préparer les données
data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
prepared_train_data <- data_prep$prepare_data(train_data)
prepared_test_data <- data_prep$prepare_data(test_data)

# Séparer les caractéristiques et la variable cible
X_train <- as.matrix(prepared_train_data[, -1])  # Retirer la première colonne (variable cible)
y_train <- train_data$Species

X_test <- as.matrix(prepared_test_data[, -1])  # Retirer la première colonne (variable cible)
y_test <- test_data$Species

# Créer une instance de la classe
model <- LogisticRegressionMultinomial$new(learning_rate = 0.01, num_iterations = 1000)

# Ajuster le modèle sur l'ensemble d'entraînement
model$fit(X_train, y_train)

# Prédire sur l'ensemble de test
predictions <- model$predict(X_test)

# Afficher les prédictions
print(predictions)

# Évaluer la performance
accuracy <- sum(predictions == as.numeric(y_test)) / length(y_test)
cat("Accuracy:", accuracy, "\n")

# Matrice de confusion pour voir la performance
confusion_matrix <- table(Predicted = predictions, Actual = as.numeric(y_test))
print(confusion_matrix)

