# nolint start
# Générer la documentation
# roxygen2::roxygenise()

# 14/11 -> Le test sur credit_card_rmd a montré que le problème vient du modèle et non du préprocessing #### OK

#' @TODO: 
#' Tester avec DeviceModel # Awa  #### OK
#' Tester avec StudentPerformance # Daniella #### A REVOIR
#' Implement the LogisticRegressionMultinomial class with Adam optimizer # Quentin #### OK
#' predict_proba() pour avoir les probabilités des classes # Daniella # A REVOIR DT sklearn
#' comparer non seulement avec nnet mais sklearn (rapport) # Quentin  #### OK
#' R shiny choisir la variable cible/explicatives # Daniella #### OK
#' Revoir le var importance(à traiter et écrire dans le rapport) # Awa #### Tester avec Iris et nnet 
#' Pouvoir choisir plusieurs fonction de perte (logistique, quadratique, etc.) # Quentin # A tester(deviance) https://eric.univ-lyon2.fr/ricco/cours/slides/logistic_regression_ml.pdf
#' Pouvoir choisir plusieurs optimiseurs (Adam, SGD, etc.) # Awa(fit) #### LaTeX
#' Pouvoir choisir plusieurs régularisations (L1, L2, ElasticNet) # Daniella # EN COURS
#' Ajouter var select # Awa + Daniella #### EN COURS  
#' 
#' Paralleliser les calculs
#' Latex formules # Quentin -> Overleaf + plan(table des matières)      #### OK
#' Exportation sur github(package) # Quentin
#' Exportation en PMML # Daniella 
#' transform_input() # Daniella
#' 
#' @NEXT
#' On va essayer d'améliorer le modèle en utilisant un Adam optimizer  # Quentin #### OK
#' IMPLEMENTER IN EARLY STOPPING avec la fonction de loss Implémenter un validation set ? Plus DataPreparer ? # Quentin
#' INCORPORER D'autres métriques(print) (F1, precision, recall, ROC AUC, etc.  probabilité d'appartenance aux classes) # Daniella
#' Faire le summary, affichier les hyperparamètres #### OK 
#' et les coefficients, nombre d'optimisation, learning rate, beta1, beta2, epsilon  (Adam), # Awa #### OK
#' Peut-être ne pas utiliser caret 
#' Sortie graphique, fonction de loss en fonction des itérations               #### OK
#' Completer le summary avec ma fonction de loss # Quentin
#' @BONUS
#' Mettre en image Docker
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

    beta1 = 0.9,  # Parameters for momentum of Adam
    beta2 = 0.999, # Parameters for second momentum of Adam
    epsilon = 1e-8, # small constant
    
    #' @description Initializes a new instance of the `LogisticRegressionMultinomial` class.
    #' @param learning_rate Numeric. Sets the learning rate for gradient descent. Default is 0.01.
    #' @param num_iterations Integer. Specifies the number of gradient descent iterations. Default is 1000.
    #' @param loss Character. Specifies the loss function to use. Options are "logistique", "quadratique", "deviance". Default is "logistique".
    #' @return A new `LogisticRegressionMultinomial` object.
    initialize = function(learning_rate = 0.01, num_iterations = 1000, loss = "logistique") {
      self$learning_rate <- learning_rate
      self$num_iterations <- num_iterations
      self$loss_history <- numeric(num_iterations)
      
      if (loss == "logistique") {
        self$loss_function <- self$log_loss
      } else if (loss == "quadratique") {
        self$loss_function <- self$mse_loss
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
      y <- factor(y)  # Convert y to factor to ensure consistent class levels
      unique_classes <- levels(y)  # Use levels of factor y      num_classes <- length(unique_classes)
      num_samples <- nrow(X)
      num_features <- ncol(X)
      num_classes <- length(unique_classes)
      # Initialize coefficients
      self$coefficients <- matrix(0, nrow = num_features + 1, ncol = num_classes)
      X <- cbind(1, X)  # Add intercept term
      
      # Variables for Adam Optimizer
      m <- matrix(0, nrow = num_features + 1, ncol = num_classes)
      v <- matrix(0, nrow = num_features + 1, ncol = num_classes)

      for (i in 1:self$num_iterations) {
        # Compute class probabilities
        linear_model <- X %*% self$coefficients
        probabilities <- self$softmax(linear_model)
        
        # Compute the gradient
        one_hot_y <- self$one_hot_encode(y, unique_classes)
        loss <- self$loss_function(one_hot_y, probabilities)
        self$loss_history[i] <- loss  # Stock the loss value
        
        cat("Iteration:", i, "Loss:", loss, "\n")

        error <- probabilities - one_hot_y
        gradient <- t(X) %*% error / num_samples

        # Adam Optimizer pour la mise à jour des coefficients
        m <- self$beta1 * m + (1 - self$beta1) * gradient
        v <- self$beta2 * v + (1 - self$beta2) * (gradient ^ 2)
        
        m_hat <- m / (1 - self$beta1 ^ i)
        v_hat <- v / (1 - self$beta2 ^ i)

        
        # Update coefficients
        self$coefficients <- self$coefficients - self$learning_rate * m_hat / (sqrt(v_hat) + self$epsilon)
        
      }
    },
    
    #' @description Computes the softmax of the input matrix.
    #' @param z A matrix of linear model outputs.
    #' @return A matrix of softmax probabilities for each class.
    softmax = function(z) {
      exp_z <- exp(z - apply(z, 1, max))  # Subtract max per row to prevent overflow
      return(exp_z / rowSums(exp_z))
    },
    
    #' @description One-hot encodes the response variable.
    #' @param y A vector representing the response variable.
    #' @param unique_classes A vector of unique class labels.
    #' @return A binary matrix where each row corresponds to a sample, and each column corresponds to a class.
    one_hot_encode = function(y, unique_classes) {
      y <- factor(y, levels = unique_classes)  # Ensure consistent class ordering
      one_hot <- matrix(0, nrow = length(y), ncol = length(unique_classes))
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
    
    #' @description This function provides a summary of the model's performance on the test data.
    #' @param X_test A matrix or data frame of test features.
    #' @param y_test A vector of true labels for the test data.
    #' @details The function generates predictions for the test data and prints a confusion matrix. It also calculates and prints performance metrics such as F1-score, precision, and recall using the caret package.
    #' @return Prints the confusion matrix and performance metrics.
    #' @examples
    #' \dontrun{
    #' model$summary(X_test, y_test)
    #' }
    #' @export
    summary = function(X_test, y_test) {
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
      
      # LOSS FUNCTIONS
      log_loss = function(y_true, y_pred) {
        epsilon <- 1e-15  # Small value to prevent log(0)
        y_pred <- pmax(pmin(y_pred, 1 - epsilon), epsilon) 
        loss <- -sum(y_true * log(y_pred))  # Régularisez par 1/N ?
        return(loss)

      },

      mse_loss = function(y_true, y_pred) {
        0.5 * mean((y_true - y_pred)^2)
      }
  )
)

# nolint end