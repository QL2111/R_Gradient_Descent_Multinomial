# Charger les fichiers de fonctions
source("R/DataPreparer.R")
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
# Vous devez spécifier 'target_col' pour indiquer la variable cible
prepared_train_data <- data_prep$prepare_data(train_data, target_col = "Species")
prepared_test_data <- data_prep$prepare_data(test_data, target_col = "Species")

# Vérifier la structure des données préparées
str(prepared_train_data)
str(prepared_test_data)

# Extraire X_train, X_test, y_train, y_test depuis la liste retournée par prepare_data
X_train <- as.matrix(prepared_train_data$X_train)
y_train <- prepared_train_data$y_train

X_test <- as.matrix(prepared_test_data$X_test)
y_test <- prepared_test_data$y_test

# Créer une instance de la classe LogisticRegressionMultinomial
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
