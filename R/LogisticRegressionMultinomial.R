library(R6)
library(glmnet)

#' @title Multinomial Logistic Regression Class
#' @description The `LogisticRegressionMultinomial` class implements multinomial logistic regression using gradient descent.
#' @details This class allows users to fit a multinomial logistic regression model, calculate class probabilities with softmax, and make predictions. It supports customization of the learning rate and the number of iterations for the gradient descent optimization.
#'
#' @field coefficients Matrix of model coefficients, initialized during the `fit` method.
#' @field learning_rate Numeric value representing the learning rate for gradient descent. Default is 0.01.
#' @field num_iterations Integer specifying the number of iterations for gradient descent. Default is 1000.
#'
#' @export
LogisticRegressionMultinomial <- R6Class("LogisticRegressionMultinomial",
                                         public = list(
                                           # Fields
                                           coefficients = NULL,         # Matrix of model coefficients
                                           learning_rate = NULL,        # Learning rate for gradient descent
                                           num_iterations = NULL,       # Number of iterations for gradient descent
                                           data_structure = NULL,       # Structure of input data
                                           
                                           #' @description Initializes a new instance of the `LogisticRegressionMultinomial` class.
                                           #' @param learning_rate Numeric. Sets the learning rate for gradient descent. Default is 0.01.
                                           #' @param num_iterations Integer. Specifies the number of gradient descent iterations. Default is 1000.
                                           initialize = function(learning_rate = 0.01, num_iterations = 1000) {
                                             self$learning_rate <- learning_rate
                                             self$num_iterations <- num_iterations
                                           },
                                           
                                           #' @description Fits the multinomial logistic regression model to the provided data.
                                           #' @param X A data frame or matrix of predictors.
                                           #' @param y A factor or character vector representing the response variable (target classes).
                                           #"""""""""""""""""""""""""""""""""""""""""""fit"""""""""""""""""""""""""""""""""""""""""""""""""
                                           fit = function(X, y) {
                                             unique_classes <- unique(y)
                                             num_classes <- length(unique_classes)
                                             num_samples <- nrow(X)
                                             num_features <- ncol(X)
                                             
                                             # Initialiser les coefficients comme une matrice de zéros
                                             self$coefficients <- matrix(0, nrow = num_features + 1, ncol = num_classes)
                                             
                                             # Ajouter une colonne de 1 pour le terme d'interception et convertir X en matrice numérique
                                             X <- as.matrix(cbind(1, X))
                                             
                                             # Enregistrer la structure des données
                                             self$data_structure <- list(
                                               n_observations = num_samples,
                                               n_features = num_features,
                                               feature_names = colnames(X)
                                             )
                                             
                                             # Descente de gradient
                                             for (i in 1:self$num_iterations) {
                                               # Calculer les prédictions linéaires
                                               linear_model <- X %*% self$coefficients
                                               probabilities <- self$softmax(linear_model)
                                               
                                               # Calculer l'erreur et le gradient
                                               error <- probabilities - self$one_hot_encode(y, unique_classes)
                                               gradient <- t(X) %*% error / num_samples
                                               
                                               # Mettre à jour les coefficients
                                               self$coefficients <- self$coefficients - self$learning_rate * gradient
                                             }
                                           },
                                           
                                           #' @description Predicts the class probabilities for new data.
                                           #' @param X A data frame or matrix of predictors.
                                           #"""""""""""""""""""""""""""""""""""""""""""predict_proba"""""""""""""""""""""""""""""""""""""""""""""""""
                                           predict_proba = function(X) {
                                             if (is.null(self$coefficients)) {
                                               stop("Model not yet fitted. Please run fit() first.")
                                             }
                                             X <- cbind(1, as.matrix(X))  # Add intercept term
                                             linear_preds <- X %*% self$coefficients
                                             proba <- self$softmax(linear_preds)
                                             return(proba)
                                           },
                                           
                                           #' @description Computes the softmax of the input matrix.
                                           #' @param z A matrix of linear model outputs.
                                           softmax = function(z) {
                                             exp_z <- exp(z - max(z))  # Stabilize by subtracting max value
                                             return(exp_z / rowSums(exp_z))
                                           },
                                           
                                           #' @description One-hot encodes the response variable.
                                           #' @param y A vector representing the response variable.
                                           #' @param unique_classes A vector of unique class labels.
                                           #' @return A matrix of one-hot encoded class labels.
                                           #"""""""""""""""""""""""""""""""""""""""""""one_hot_encode"""""""""""""""""""""""""""""""""""""""""""""""""
                                           one_hot_encode = function(y, unique_classes) {
                                             one_hot <- matrix(0, nrow = length(y), ncol = length(unique_classes))
                                             for (i in seq_along(y)) {
                                               one_hot[i, which(unique_classes == y[i])] <- 1
                                             }
                                             return(one_hot)
                                           },
                                           
                                           #' @description Predicts the class labels for new data.
                                           #' @param X A data frame or matrix of predictors.
                                           #"""""""""""""""""""""""""""""""""""""""""""predict"""""""""""""""""""""""""""""""""""""""""""""""""
                                           predict = function(X) {
                                             probabilities <- self$predict_proba(X)
                                             return(apply(probabilities, 1, which.max))
                                           },
                                           
                                           #' @description Displays a summary of the model's fitted parameters.
                                           #' @details This method prints the model coefficients and data structure.
                                           #' @return Prints the model summary.
                                           
                                           #"""""""""""""""""""""""""""""""""""""""""""summary"""""""""""""""""""""""""""""""""""""""""""""""""
                                           summary = function() {
                                             if (is.null(self$coefficients)) {
                                               cat("Model not yet fitted. Please run fit() first.\n")
                                               return()
                                             }
                                             cat("=== Model Summary ===\n")
                                             cat("\nCoefficients:\n")
                                             print(self$coefficients)
                                             cat("\nData Structure:\n")
                                             cat("Number of observations:", self$data_structure$n_observations, "\n")
                                             cat("Number of predictors:", self$data_structure$n_features, "\n")
                                             cat("Predictor names:", paste(self$data_structure$feature_names, collapse = ", "), "\n")
                                           },
                                           
                                           #' @description Selects important features based on coefficient magnitude.
                                           #' @param X A data frame of predictors.
                                           #' @param y A factor or character vector representing the response variable.
                                           #' @param threshold Numeric value for the coefficient magnitude threshold. Default is 0.1.
                                           
                                           #"""""""""""""""""""""""""""""""""""""""""""var_select"""""""""""""""""""""""""""""""""""""""""""""""""
                                           
                                           var_select = function(X, y, threshold = 0.1) {
                                             # Convertir X en matrice numérique et y en format binaire
                                             X_matrix <- as.matrix(X)
                                             y_binary <- as.numeric(y == levels(y)[2])  # Transformer y en binaire pour glmnet
                                             
                                             # Ajuster un modèle avec régularisation L1 (lasso) pour la sélection de variables
                                             cv_model <- cv.glmnet(X_matrix, y_binary, alpha = 1, family = "binomial")
                                             
                                             # Extraire les coefficients non nuls avec le meilleur lambda (pénalisation optimale)
                                             best_lambda <- cv_model$lambda.min
                                             coefficients <- as.vector(coef(cv_model, s = best_lambda))
                                             selected_vars <- names(coefficients)[abs(coefficients) >= threshold]
                                             
                                             # Filtrer les variables sélectionnées dans les données
                                             selected_data <- X[, selected_vars, drop = FALSE]
                                             
                                             cat("Variables sélectionnées basées sur l'importance prédictive :\n")
                                             print(selected_vars)
                                             
                                             return(selected_data)
                                           },
                                           
                                           #' @description Displays model details.
                                           
                                           #"""""""""""""""""""""""""""""""""""""""""""print"""""""""""""""""""""""""""""""""""""""""""""""""
                                           print = function() {
                                             cat("=== Multinomial Logistic Regression Model ===\n")
                                             if (is.null(self$coefficients)) {
                                               cat("Model not yet fitted. Please run fit() first.\n")
                                             } else {
                                               cat("\nAdjusted Parameters:\n")
                                               print(self$coefficients)
                                               cat("\nData Structure:\n")
                                               cat("Number of observations:", self$data_structure$n_observations, "\n")
                                               cat("Number of predictors:", self$data_structure$n_features, "\n")
                                               cat("Predictor names:", paste(self$data_structure$feature_names, collapse = ", "), "\n")
                                             }
                                           }
                                        )
)

