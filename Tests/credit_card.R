# https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data Variable à predire: Industry

# Charger les bibliothèques nécessaires
library(R6)

# Charger les fichiers de fonctions
source("R/DataPreparer.R")
source("R/factor_analysis_mixed.R")
source("R/LogisticRegressionMultinomial.R")

print("\nCredit Card Approval Prediction Example \n")

# Charger le jeu de données depuis un fichier local après téléchargement de Kaggle
data_path <- "data/credit_card.csv"  # Remplacez par le chemin de votre fichier
data <- read.csv(data_path)

# S'assurer que la variable cible est un facteur
data$Industry <- as.factor(data$Industry)

# Diviser les données en ensembles d'entraînement et de test
set.seed(42)  # Pour la reproductibilité
train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))  # 70% pour l'entraînement
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Séparer les caractéristiques et la variable cible
X_train <- train_data[, -which(names(train_data) == "Industry")]
y_train <- train_data$Industry

X_test <- test_data[, -which(names(test_data) == "Industry")]
y_test <- test_data$Industry

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
model <- LogisticRegressionMultinomial$new(learning_rate = 0.1, num_iterations = 1000)
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
