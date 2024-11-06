# Charger les fichiers de fonctions
source("R/DataPreparer.R")
source("R/factor_analysis_mixed.R")
source("R/LogisticRegressionMultinomial.R")

# Charger les données Iris
data(iris)
iris$Species <- as.factor(iris$Species)

# Diviser les données en train/test avant le traitement
set.seed(42)  # Pour la reproductibilité
train_indices <- sample(1:nrow(iris), size = 0.7 * nrow(iris))  # 70% pour l'entraînement
train_data <- iris[train_indices, ]
test_data <- iris[-train_indices, ]

# Séparer les caractéristiques et la variable cible avant traitement
X_train <- train_data[, -which(names(train_data) == "Species")]
y_train <- train_data$Species

X_test <- test_data[, -which(names(test_data) == "Species")]
y_test <- test_data$Species

# Préparer les données sans inclure la variable cible
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
model <- LogisticRegressionMultinomial$new(learning_rate = 0.01, num_iterations = 1000)
model$fit(X_train_matrix, y_train_numeric)

# Prédire sur l'ensemble de test
predictions <- model$predict(X_test_matrix)

# Afficher les prédictions
print(predictions)

# Calculer et afficher l'accuracy
accuracy <- sum(predictions == y_test_numeric) / length(y_test_numeric)
cat("Accuracy:", accuracy, "\n")

# Matrice de confusion pour évaluer les performances
confusion_matrix <- table(Predicted = predictions, Actual = y_test_numeric)
print(confusion_matrix)

# Importance des variables
model$var_importance()