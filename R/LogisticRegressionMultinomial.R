# nolint start
# Générer la documentation
# roxygen2::roxygenise()

# 14/11 -> Le test sur credit_card_rmd a montré que le problème vient du modèle et non du préprocessing #### OK

#' @TODO: 
#' Tester avec StudentPerformance # Daniella #### A REVOIR
#' predict_proba() pour avoir les probabilités des classes + ajouter au print # Daniella # A REVOIR DT sklearn
#' Revoir le var importance(à traiter et écrire dans le rapport) # Awa #### Tester avec Iris et nnet 
#' Pouvoir choisir plusieurs optimiseurs (Adam, SGD, etc.) # Awa(fit) #### LaTeX SGD pas efficace ?
#' Pouvoir choisir plusieurs régularisations (L1, L2, ElasticNet) # Daniella # EN COURS
#' Ajouter var select # Awa + Daniella #### EN COURS  
#' Analyse Factorielle (Plus de dimension)
#' One hot encoding one vs one et one vs all et multinomial native
#' Paralleliser les calculs
#' Exportation sur github(package) # Quentin
#' Exportation en PMML # Daniella 
#' transform_input() # Daniella
#' 
#' 
#' @NEXT
#' IMPLEMENTER IN EARLY STOPPING avec la fonction de loss Implémenter un validation set ? Plus DataPreparer ? # Quentin
#' INCORPORER D'autres métriques(print) (F1, precision, recall, ROC AUC, etc.  probabilité d'appartenance aux classes) # Daniella
#' Peut-être ne pas utiliser caret 
#' @BONUS
#' Mettre en image Docker
#' 
#' @DONE
#' #' Tester avec DeviceModel # Awa  #### OK
#' #' Implement the LogisticRegressionMultinomial class with Adam optimizer # Quentin #### OK
#' #' comparer non seulement avec nnet mais sklearn (rapport) # Quentin  #### OK
#' #' R shiny choisir la variable cible/explicatives # Daniella #### OK
#' Pouvoir choisir plusieurs fonction de perte (logistique, quadratique, etc.) # Quentin # A tester(deviance) https://eric.univ-lyon2.fr/ricco/cours/slides/logistic_regression_ml.pdf
#' On va essayer d'améliorer le modèle en utilisant un Adam optimizer  # Quentin #### OK
#' Latex formules # Quentin -> Overleaf + plan(table des matières)      #### OK
#' Sortie graphique, fonction de loss en fonction des itérations               #### OK
#' Completer le summary avec ma fonction de loss # Quentin #### OK
#'  nombre d'optimisation, learning rate, beta1, beta2, epsilon  (Adam), # Awa #### OK
#' Faire le summary, affichier les hyperparamètres #### OK 
#' 
#' 




#' @title Logistic Regression Multinomial Class
#' @description The `LogisticRegressionMultinomial` class implements multinomial logistic regression using gradient descent and the Adam optimizer.
#' @details This class allows users to fit a multinomial logistic regression model, calculate class probabilities using softmax, and make predictions. It supports features like loss tracking, variable importance calculation, and a summary of model performance.
#'
#' @field coefficients Matrix of model coefficients, initialized during the `fit` method.
#' @field learning_rate Numeric. Learning rate for gradient descent optimization. Default is 0.01.
#' @field num_iterations Integer. Number of iterations for gradient descent optimization. Default is 1000.
#' @field loss_history Numeric vector. Tracks the loss at each iteration during training.
#' @field beta1 Numeric. Momentum parameter for Adam optimizer. Default is 0.9.
#' @field beta2 Numeric. Second momentum parameter for Adam optimizer. Default is 0.999.
#' @field epsilon Numeric. Small constant for numerical stability in Adam optimizer. Default is 1e-8.
#'
#' @export
LogisticRegressionMultinomial <- R6Class("LogisticRegressionMultinomial",
  public = list(
    coefficients = NULL,
    learning_rate = NULL,
    num_iterations = NULL,
    loss_history = NULL,  # Stock the loss values for each iteration
    loss_function = NULL,  # Loss function to use
    loss_name = NULL,  # Name of the loss function

    optimizer = NULL, # Optimizer to use
    beta1 = NULL,  # Parameters for momentum of Adam
    beta2 = NULL, # Parameters for second momentum of Adam
    epsilon = NULL, # small constant
    
    #' @description Initializes a new instance of the `LogisticRegressionMultinomial` class.
    #' @param learning_rate Numeric. Sets the learning rate for gradient descent. Default is 0.01.
    #' @param num_iterations Integer. Specifies the number of gradient descent iterations. Default is 1000.
    #' @param loss Character. Specifies the loss function to use. Options are "logistique", "quadratique", "deviance". Default is "logistique".
    #' @return A new `LogisticRegressionMultinomial` object.
    initialize = function(learning_rate = 0.01, num_iterations = 1000, loss = "logistique", optimizer = "adam", beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
      self$learning_rate = learning_rate
      self$num_iterations = num_iterations
      self$loss_history = numeric(num_iterations)
      self$optimizer = optimizer
      self$loss_name = loss  # Store the name of the loss function
      self$beta1 = beta1
      self$beta2 = beta2
      self$epsilon = epsilon
      
      
      if (loss == "logistique") {
        self$loss_function = self$log_loss
      } else if (loss == "quadratique") {
        self$loss_function = self$mse_loss
      } else {
        stop("Fonction de perte non reconnue")
      }
    },

    
    #' @description Fits the multinomial logistic regression model to the provided data.
    #' @param X A data frame or matrix of predictors (features), where rows represent samples and columns represent features.
    #' @param y A factor or character vector representing the response variable (target classes).
    #' @details The `fit` method initializes model coefficients and applies gradient descent to minimize the loss function. It calculates class probabilities with softmax and updates coefficients based on the gradient.
    #' @return No return value; updates the model's coefficients.
    fit = function(X, y) {
      y = factor(y)  # Convert y to factor to ensure consistent class levels
      unique_classes = levels(y)  # Use levels of factor y      num_classes <- length(unique_classes)
      num_samples = nrow(X)
      num_features = ncol(X)
      num_classes = length(unique_classes)
      # Initialize coefficients
      self$coefficients = matrix(0, nrow = num_features + 1, ncol = num_classes)
      X = cbind(1, X)  # Add intercept term
      
      if (self$optimizer == "adam") {
        self$adam_optimizer(X, y, unique_classes, num_samples, num_features, num_classes)
      } else if (self$optimizer == "sgd") {
        self$sgd_optimizer(X, y, unique_classes, num_samples, num_features, num_classes)
      } else {
        stop("Optimiseur non reconnu")
      }
    },

    # Code AWA -> Performance du SGD faible 
    # fit = function(X, y) {
    #   y <- factor(y)
    #   unique_classes <- levels(y)
    #   num_classes <- length(unique_classes)
    #   num_samples <- nrow(X)
    #   num_features <- ncol(X)
      
    #   # Initialisation des coefficients
    #   self$coefficients <- matrix(0, nrow = num_features + 1, ncol = num_classes)
    #   X <- cbind(1, X)
      
    #   # Variables pour Adam
    #   m <- matrix(0, nrow = num_features + 1, ncol = num_classes)
    #   v <- matrix(0, nrow = num_features + 1, ncol = num_classes)
      
    #   for (i in 1:self$num_iterations) {
    #     linear_model <- X %*% self$coefficients
    #     probabilities <- self$softmax(linear_model)
    #     one_hot_y <- self$one_hot_encode(y, unique_classes)
    #     loss <- -sum(one_hot_y * log(probabilities)) / num_samples
    #     self$loss_history[i] <- loss
    #     cat("Iteration:", i, "Loss:", loss, "\n")
        
    #     error <- probabilities - one_hot_y
    #     gradient <- t(X) %*% error / num_samples
        
    #     if (self$optimizer == "adam") {
    #       # Mise à jour avec Adam
    #       m <- self$beta1 * m + (1 - self$beta1) * gradient
    #       v <- self$beta2 * v + (1 - self$beta2) * (gradient ^ 2)
    #       m_hat <- m / (1 - self$beta1 ^ i)
    #       v_hat <- v / (1 - self$beta2 ^ i)
    #       self$coefficients <- self$coefficients - self$learning_rate * m_hat / (sqrt(v_hat) + self$epsilon)
    #     } else if (self$optimizer == "sgd") {
          
    #       # Mise à jour avec SGD
    #       self$coefficients <- self$coefficients - self$learning_rate * gradient
    #     } else {
    #       stop("Unsupported optimizer: ", self$optimizer)
    #     }
    #   }
    # },
    
    #' @description Computes the softmax of the input matrix.
    #' @param z A matrix of linear model outputs.
    #' @return A matrix of softmax probabilities for each class.
    softmax = function(z) {
      exp_z = exp(z - apply(z, 1, max))  # Subtract max per row to prevent overflow
      return(exp_z / rowSums(exp_z))
    },
    
    #' @description One-hot encodes the response variable.
    #' @param y A vector representing the response variable.
    #' @param unique_classes A vector of unique class labels.
    #' @return A binary matrix where each row corresponds to a sample, and each column corresponds to a class.
    one_hot_encode = function(y, unique_classes) {
      y = factor(y, levels = unique_classes)  # Ensure consistent class ordering
      one_hot = matrix(0, nrow = length(y), ncol = length(unique_classes))
      for (i in 1:length(y)) {
        one_hot[i, as.integer(y[i])] <- 1
      }
      return(one_hot)
    },

    
    #' @description Predicts the class labels for new data.
    #' @param X A data frame or matrix of predictors, where rows are samples and columns are features.
    #' @return A vector of predicted class labels for each sample.
    predict = function(X) {
      X <- cbind(1, X)  # Add intercept term
      linear_model <- X %*% self$coefficients
      probabilities <- self$softmax(linear_model)
      return(apply(probabilities, 1, which.max))  # Convert back to 0 and 1 instead of 1 and 2
    },
    
    #' @description This function calculates the importance of each feature based on the absolute value of the coefficients.
    #' @return A vector of feature importance scores, sorted in descending order.
    #' @examples
    #' \dontrun{
    #' model$var_importance()
    #' }
    #' @export
    var_importance = function() {
      coef_matrix <- abs(self$coefficients[-1, ])  # Exclude intercept term
      importance_scores <- rowSums(coef_matrix)    # Sum of absolute coefficients for each feature
      importance_ranked <- sort(importance_scores, decreasing = TRUE)
      
      # Affichage des importances
      cat("Variable Importance (sorted):\n")
      for (i in 1:length(importance_ranked)) {
        cat(names(importance_ranked)[i], ": ", importance_ranked[i], "\n")
      }
      
      return(importance_ranked)
    },


    #' @description This function plots the loss history to visualize the convergence of the loss function over iterations.
    #' @details The function checks if the loss history is available and non-empty. If the loss history is empty, it stops and prompts the user to run the 'fit' method first. Otherwise, it plots the loss history.
    #' @return A plot showing the convergence of the loss function over iterations.
    #' @examples
    #' \dontrun{
    #' model$plot_loss()
    #' }
    #' @export 
    plot_loss = function() {
      # Check if we have loss history
      if (is.null(self$loss_history) || length(self$loss_history) == 0) {
        stop("Loss history is empty. Please run the 'fit' method first.")
      }
      
      # plot loss history
      plot(self$loss_history, type = "l", col = "blue", lwd = 2,
           main = "Loss Function Convergence",
           xlab = "Iterations", ylab = "Loss")
    },
    
    #' @description Displays the hyperparameters of the trained model.
    #' @details This function prints out the hyperparameters of the model, including the optimizer, learning rate, number of iterations, and loss function. If the optimizer is "adam", it also prints out the Adam-specific parameters: Beta1, Beta2, and Epsilon.
    #' @return None. This function is used for its side effect of printing the model's hyperparameters.
    #' @examples
    #' \dontrun{
    #' model$summary()
    #' }
    # Paramètres du modèle
    summary = function() {
      

      # En attendant d'avoir le print
      # Model Hyperparameters
      cat("\n=== Model Hyperparameters ===\n")
      cat("Optimizer: ", self$optimizer, "\n")
      cat("Learning Rate: ", self$learning_rate, "\n")
      cat("Number of Iterations: ", self$num_iterations, "\n")
      cat("Loss Function: ", self$loss_name, "\n")
      if (self$optimizer == "adam") {
        cat("Beta1 (Adam): ", self$beta1, "\n")
        cat("Beta2 (Adam): ", self$beta2, "\n")
        cat("Epsilon (Adam): ", self$epsilon, "\n")
      }
    },

    #' @description This function prints the results of the logistic regression multinomial model on the test data.
    #' @param X_test A data frame or matrix containing the test features.
    #' @param y_test A vector containing the true labels for the test data.
    #'
    #' @return This function does not return a value. It prints the confusion matrix, classification report, and F1 score.
    #'
    #' @details
    #' The function performs the following steps:
    #' \itemize{
    #'   \item Predicts the labels for the test data using the model.
    #'   \item Computes and prints the confusion matrix.
    #'   \item Computes and prints the classification report using the `caret` package.
    #'   \item Computes and prints the weighted F1 score using the `MLmetrics` package.
    #' }
    #'
    #' @examples
    #' \dontrun{
    #' model$print(X_test, y_test)
    #' }
    #'
    #' @import caret
    #' @import MLmetrics
    #' @export
    print = function(X_test, y_test) {
      predictions <- self$predict(X_test)
      
      # Confusion Matrix
      confusion_matrix <- table(Predicted = predictions, Actual = y_test)
      print("Confusion Matrix:")
      print(confusion_matrix)
      
      #  F1-score, precision, Recall
      library(caret)
      library(MLmetrics)
      report <- confusionMatrix(as.factor(predictions), as.factor(y_test))
      print(report)
      
      f1_weighted <- F1_Score(y_pred = predictions, y_true = y_test)
      cat("F1 Score:", f1_weighted, "\n")
    },
    
      
      

    #' @description This function computes the log loss, also known as logistic loss or cross-entropy loss, 
    #' between the true labels and the predicted probabilities.
    #'
    #' @param y_true A numeric vector of true labels.
    #' @param y_pred A numeric vector of predicted probabilities.
    #' @return A numeric value representing the log loss.
    #' @examples
    #' y_true <- c(1, 0, 1, 0)
    #' y_pred <- c(0.9, 0.1, 0.8, 0.2)
    #' log_loss(y_true, y_pred)
    #' @export
    # LOSS FUNCTIONS
    log_loss = function(y_true, y_pred) {
      epsilon <- 1e-15  # Small value to prevent log(0)
      y_pred <- pmax(pmin(y_pred, 1 - epsilon), epsilon) 
      loss <- -sum(y_true * log(y_pred))  # Régularisez par 1/N ?
      return(loss)

    },

    
    #' @description This function computes the mean squared error (MSE) loss between the true labels 
    #' and the predicted values.
    #'
    #' @param y_true A numeric vector of true labels.
    #' @param y_pred A numeric vector of predicted values.
    #' @return A numeric value representing the MSE loss.
    #' @examples
    #' y_true <- c(1, 0, 1, 0)
    #' y_pred <- c(0.9, 0.1, 0.8, 0.2)
    #' mse_loss(y_true, y_pred)
    #' @export
    #'
    mse_loss = function(y_true, y_pred) {
      0.5 * mean((y_true - y_pred)^2)
    },

    # mse_loss = function(y_true, y_pred) { # MSE MULTINOMIALE?
    #   0.5 * sum((y_true - y_pred)^2)
    # }


    #' @description Adam optimizer for updating coefficients.
    #' @param X Matrix of predictors with intercept term added.
    #' @param y Factor vector of response variable.
    #' @param unique_classes Vector of unique class labels.
    #' @param num_samples Number of samples.
    #' @param num_features Number of features.
    #' @param num_classes Number of classes.
    adam_optimizer = function(X, y, unique_classes, num_samples, num_features, num_classes) {
      m <- matrix(0, nrow = num_features + 1, ncol = num_classes)
      v <- matrix(0, nrow = num_features + 1, ncol = num_classes)

      for (i in 1:self$num_iterations) {
        linear_model <- X %*% self$coefficients
        probabilities <- self$softmax(linear_model)
        one_hot_y <- self$one_hot_encode(y, unique_classes)
        loss <- self$loss_function(one_hot_y, probabilities)
        self$loss_history[i] <- loss
        
        cat("Iteration:", i, "Loss:", loss, "\n")

        error <- probabilities - one_hot_y
        gradient <- t(X) %*% error / num_samples

        m <- self$beta1 * m + (1 - self$beta1) * gradient
        v <- self$beta2 * v + (1 - self$beta2) * (gradient ^ 2)
        
        m_hat <- m / (1 - self$beta1 ^ i)
        v_hat <- v / (1 - self$beta2 ^ i)

        self$coefficients <- self$coefficients - self$learning_rate * m_hat / (sqrt(v_hat) + self$epsilon)
      }
    },

    #' @description SGD optimizer for updating coefficients.
    #' @param X Matrix of predictors with intercept term added.
    #' @param y Factor vector of response variable.
    #' @param unique_classes Vector of unique class labels.
    #' @param num_samples Number of samples.
    #' @param num_features Number of features.
    #' @param num_classes Number of classes.
    sgd_optimizer = function(X, y, unique_classes, num_samples, num_features, num_classes) {
      for (i in 1:self$num_iterations) {
        linear_model <- X %*% self$coefficients
        probabilities <- self$softmax(linear_model)
        one_hot_y <- self$one_hot_encode(y, unique_classes)
        loss <- self$loss_function(one_hot_y, probabilities)
        self$loss_history[i] <- loss
        
        cat("Iteration:", i, "Loss:", loss, "\n")

        error <- probabilities - one_hot_y
        gradient <- t(X) %*% error / num_samples

        self$coefficients <- self$coefficients - self$learning_rate * gradient
      }
    }
  )
)

# nolint end