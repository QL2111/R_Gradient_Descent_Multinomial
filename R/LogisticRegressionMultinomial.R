# LogisticRegressionMultinomial.R

LogisticRegressionMultinomial <- R6Class("LogisticRegressionMultinomial",
  public = list(
    coefficients = NULL,
    learning_rate = NULL,
    num_iterations = NULL,
    
    initialize = function(learning_rate = 0.01, num_iterations = 1000) {
      self$learning_rate <- learning_rate
      self$num_iterations <- num_iterations
    },
    
    fit = function(X, y) {
      unique_classes <- unique(y)
      num_classes <- length(unique_classes)
      num_samples <- nrow(X)
      num_features <- ncol(X)
      
      self$coefficients <- matrix(0, nrow = num_features + 1, ncol = num_classes)
      X <- cbind(1, X)
      
      for (i in 1:self$num_iterations) {
        linear_model <- X %*% self$coefficients
        probabilities <- self$softmax(linear_model)
        
        error <- probabilities - self$one_hot_encode(y, unique_classes)
        gradient <- t(X) %*% error / num_samples
        
        self$coefficients <- self$coefficients - self$learning_rate * gradient
      }
    },
    
    softmax = function(z) {
      exp_z <- exp(z - max(z))
      return(exp_z / rowSums(exp_z))
    },
    
    one_hot_encode = function(y, unique_classes) {
      one_hot <- matrix(0, nrow = length(y), ncol = length(unique_classes))
      for (i in 1:length(y)) {
        one_hot[i, which(unique_classes == y[i])] <- 1
      }
      return(one_hot)
    },
    
    predict = function(X) {
      X <- cbind(1, X)
      linear_model <- X %*% self$coefficients
      probabilities <- self$softmax(linear_model)
      return(apply(probabilities, 1, which.max))
    }
  )
)
