# nolint start
# https://www.kaggle.com/datasets/lainguyn123/student-performance-factors Variable à predire: Access_to_Resources

# Charger les bibliothèques nécessaires
library(R6)

# Charger les fichiers de fonctions
source("R/DataPreparer.R")
source("R/LogisticRegressionMultinomial.R")

# Charger le jeu de données après téléchargement de Kaggle
data_path <- "Data/user_behavior_dataset.csv"  # Remplacez par le chemin de votre fichier
data <- read.csv(data_path)

# S'assurer que la variable cible est un facteur
data$User.Behavior.Class <- as.factor(data$User.Behavior.Class)

# Diviser les données en ensembles d'entraînement et de test
set.seed(42)  # Pour la reproductibilité
train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))  # 70% pour l'entraînement
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Séparer les caractéristiques et la variable cible
X_train <- train_data[, -which(names(train_data) == "User.Behavior.Class")]
y_train <- train_data$User.Behavior.Class

X_test <- test_data[, -which(names(test_data) == "User.Behavior.Class")]
y_test <- test_data$User.Behavior.Class

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
model <- LogisticRegressionMultinomial$new(learning_rate = 0.1, num_iterations = 1000, loss="logistique", optimizer="adam", use_early_stopping=TRUE, regularization = "lasso")
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

#=========================test avec nnet======================
# Initialiser le modèle de régression logistique multinomial
multinom_model <- nnet::multinom(y_train_numeric ~ ., data = as.data.frame(X_train_matrix))

# Faire des prédictions sur l'ensemble de test
multinom_predictions <- predict(multinom_model, newdata = as.data.frame(X_test_matrix))

# Vérifier les prédictions
print("Multinomial Model Predictions:")
print(multinom_predictions)

# Calculer et afficher l'accuracy
multinom_accuracy <- sum(multinom_predictions == y_test_numeric) / length(y_test_numeric)
cat("Multinomial Model Accuracy nnet:", multinom_accuracy, "\n")
#======================test export pmml ==============================
#model$export_pmml("logistic_model.pmml")


# nolint end
