# @TODO: Implement the LogisticRegressionMultinomial class with Adam optimizer
# 14/11 -> Le test sur credit_card_rmd a montré que le problème vient du modèle et non du préprocessing
# On va essayer d'améliorer le modèle en utilisant un Adam optimizer
# Faire le print, affichier les hyperparamètres, et les coefficients, nombre d'optimisation, learning rate, beta1, beta2, epsilon (Adam)


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
    #' @field coefficients Matrix. Stores the model's learned coefficients.
    coefficients = NULL,
    
    #' @field learning_rate Numeric. Learning rate for the gradient descent optimization.
    learning_rate = NULL,
    
    #' @field num_iterations Integer. Number of iterations for gradient descent optimization.
    num_iterations = NULL,

    beta1 = 0.9,  # Parameters for momentum of Adam
    beta2 = 0.999, # Parameters for second momentum of Adam
    epsilon = 1e-8, # small constant
    
    #' @description Initializes a new instance of the `LogisticRegressionMultinomial` class.
    #' @param learning_rate Numeric. Sets the learning rate for gradient descent. Default is 0.01.
    #' @param num_iterations Integer. Specifies the number of gradient descent iterations. Default is 1000.
    #' @return A new `LogisticRegressionMultinomial` object.
    initialize = function(learning_rate = 0.01, num_iterations = 1000) {
      self$learning_rate <- learning_rate
      self$num_iterations <- num_iterations
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
        loss <- -sum(one_hot_y * log(probabilities)) / num_samples
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
    }

    
    #' @description Predicts the class labels for new data.
    #' @param X A data frame or matrix of predictors, where rows are samples and columns are features.
    #' @return A vector of predicted class labels for each sample.
    predict = function(X) {
      X <- cbind(1, X)  # Add intercept term
      linear_model <- X %*% self$coefficients
      probabilities <- self$softmax(linear_model)
      return(apply(probabilities, 1, which.max))
    },
    
    # Method to calculate variable importance
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
    }
  )
)
