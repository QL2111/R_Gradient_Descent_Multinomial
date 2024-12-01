# nolint start
# Générer la documentation
# roxygen2::roxygenise()

# ==============================================================TODO=====================================================
#' Rshiny -> Utiliser une librairie, retaper
#' Pouvoir choisir plusieurs régularisations (L1, L2, ElasticNet) # Daniella # EN COURS, il faut tester avec un jeu de donnée plus dur, car sur student performance, le F1 est déjà à 1

#' Faire mini batch # Quentin (Descente de gradient)
#' Sortie graphique var importances(barplot) # Awa
#' Pseudo code # Awa
#' ReadMe Github  # Quentin
#' Formulaire Shiny, rajouter l'option d'analyse factorielle et de régularisation + early stopping # Daniella
#' SMOTE # Quentin
#' Imputation par KNN ? # Quentin -> Inclure dans le rapport discussion, jeu de données lourd
#' Documentation
#' Peut-être ne pas utiliser caret() + MLmetrics + pROC +  
#' LaTeX # Awa
#' 
#' 
#' #' revoir SGD
#' #' FIT REGRESSION LOGISTIQUE VOIR STRATEGIE Mini Batch(nb paramètre de l'algorithme) au lieu de Batch Gradient Descent(Tout l'ensemble de données) 
#' ==============================================================BONUS=====================================================
#' Améliorer SGD Optimizer # Awa #### OK
#' Implémenter des objets pertinents que le model peut retourner
#' #' Paralleliser les calculs
#' #' R Shiny -> Ajouter nouveaux champ pour les hyperparamètres du modèles,  #### EN COURS + de champs possibles ?

#' 
#' ==============================================================DONE=====================================================
#' #' Test Package # Awa -> Quentin #### OK
#' #' #' Outliers ? #Quentin ### OK
#' #' help # Awa -> Quentin #### OK
#' #' Mettre en image Docker # Awa #### OK
#'Mettre un Imputer sur le datapreparer, Missing values aussi à mettre dans le datapreparer et outliers avant le scaler # Quentin ### OK
#' #' Ajouter var select # Awa #### à tester - Quentin #### OK -> pas de différences avec var importance ? 
#' #' Incorporer AFDM dans data preparer # Quentin  ncp pour le nombre de dimensions à garder(variables explicatives cumulé>95%) # Quentin #### OK MAIS accuracy faible pour student performance
#' #' Exportation en PMML # Daniella ### OK
#' #' Analyse Factorielle (Plus de dimension) # Quentin ### OK
#' #' Ajouter régularisation + export PMML dans LogisticRegressionMultinomial dans LogistRegression.R # Quentin #### OK
#' #' Implémenter analyse factorielle dans le datapreparer + tester avec studentperformance # Quentin   #### OK
#' #' Device model mauvais test -> essayer avec une autre variable cible(User Behavior classification pour voir si l'accuracy monte) # Awa #### OK
#' #' Tester Analyse factorielle multiclass tester avec student_performancce + Iris + JEU DE DONNEES avec beaucoup de col # Awa Iris + StudentPerformance # OK
#' #' intégrer le train/test split dans le datapreparer  + stratify # Quentin ### OK
#' #' INCORPORER D'autres métriques(print) (F1, precision, recall, ROC AUC, etc.  probabilité d'appartenance aux classes) # Quentin #### OK
#' #' AUC ? -> print + shiny # Quentin ####ok
#' #' Pouvoir choisir plusieurs optimiseurs (Adam, SGD, etc.) # Awa(fit) #### LaTeX SGD pas efficace ?
#' Tester var_importance et comparer avec sklearn # Quentin         #### OK
#' #' predict_proba() pour avoir les probabilités des classes + ajouter au summary # Quentin #### OK  (fait avant Daniella pour les AUC) 
#' Factoriser code factor_analysis dans DataPreparer # Quentin ### OK
#' #' Tester avec DeviceModel # Awa  #### OK
#' #' Revoir le var importance(à traiter et écrire dans le rapport) # Awa #### Tester avec Iris et nnet  #### OK
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
#' IMPLEMENTER IN EARLY STOPPING avec la fonction de loss Implémenter un validation set ? Plus DataPreparer ? # Quentin #### OK
#' Ajouter une condition pour l'early stopping, peu de données, pas bien de faire un validation set # Quentin #### OK
#' #' Tester avec StudentPerformance # Daniella Quentin OK #### 



#' #' Exportation sous forme de package R # Quentin  
#' #### OK devtools::build() 
#' Pour l'installer
#' install.packages("mon_package_0.1.0.tar.gz", repos = NULL, type = "source") 
#' installer avec github
#' devtools::install_github("Lien du repo")
#' # documentation roxygen2::roxygenise().


#' @title Logistic Regression Multinomial Class
#' @description The `LogisticRegressionMultinomial` class implements multinomial logistic regression using gradient descent and the Adam optimizer.
#' @details This class allows users to fit a multinomial logistic regression model, calculate class probabilities using softmax, and make predictions. It supports features like loss tracking, variable importance calculation, and a summary of model performance.
#' @import R6
#' 
#' @field coefficients Matrix of model coefficients, initialized during the `fit` method.
#' @field learning_rate Numeric. Learning rate for gradient descent optimization. Default is 0.01.
#' @field num_iterations Integer. Number of iterations for gradient descent optimization. Default is 1000.
#' @field loss_history Numeric vector. Tracks the loss at each iteration during training.
#' @field beta1 Numeric. Momentum parameter for Adam optimizer. Default is 0.9.
#' @field beta2 Numeric. Second momentum parameter for Adam optimizer. Default is 0.999.
#' @field epsilon Numeric. Small constant for numerical stability in Adam optimizer. Default is 1e-8.
#' @field use_early_stopping Logical. Whether to use early stopping based on validation loss. Default is TRUE.
#' @field patience Integer. Number of iterations to wait for improvement before stopping early. Default is 20.
#' @field regularization Character. Regularization method to use. Options are "none", "ridge", "lasso", "elasticnet". Default is "none".
#' @field loss_function Function. Loss function to use for optimization. Options are "quadratique", "logistique". Default is "logistique".
#' @field loss_name Character. Name of the loss function used.
#' @field optimizer Character. Optimizer to use for gradient descent. Options are "adam", "sgd". Default is "adam".
#' @field batch_size Integer. Size of the mini-batch for gradient descent. Default is 32.
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

    use_early_stopping = NULL, # Use early stopping
    patience = NULL, # Early stopping patience

    regularization = NULL, # Regularization to use
    batch_size = NULL, # Size of the mini-batch for gradient descent
    
    # class_labels = NULL,  # Store the class labels to rename them later
    
    #' @description Initializes a new instance of the `LogisticRegressionMultinomial` class.
    #' @param learning_rate Numeric. Sets the learning rate for gradient descent. Default is 0.01.
    #' @param num_iterations Integer. Specifies the number of gradient descent iterations. Default is 1000.
    #' @param loss Character. Specifies the loss function to use. Options are "logistique", "quadratique", "deviance". Default is "logistique".
    #' @param optimizer Character. Specifies the optimizer to use. Options are "adam", "sgd". Default is "adam".
    #' @param use_early_stopping Logical. Whether to use early stopping. Default is TRUE.
    #' @param patience Integer. Number of iterations to wait for improvement before stopping early. Default is 10.
    #' @param beta1 Numeric. Momentum parameter for Adam optimizer. Default is 0.9.
    #' @param beta2 Numeric. Second momentum parameter for Adam optimizer. Default is 0.999.
    #' @param epsilon Numeric. Small constant for numerical stability in Adam optimizer. Default is 1e-8.
    #' @param batch_size Integer. Size of the mini-batch for gradient descent. Default is 32, put 1 for online learning.
    #' @param regularization Character. Regularization method to use. Options are "none", "ridge", "lasso", "elasticnet". Default is "none".
    #' @return A new `LogisticRegressionMultinomial` object.
    initialize = function(learning_rate = 0.01, num_iterations = 1000, loss = "logistique", 
    optimizer = "adam", beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, patience = 20, 
    use_early_stopping = TRUE, regularization = "none", batch_size = 32) 
    {
      self$learning_rate = learning_rate
      self$num_iterations = num_iterations
      self$loss_history = numeric(num_iterations)
      self$optimizer = optimizer
      self$batch_size = batch_size  # Add batch_size to the class
      self$loss_name = loss  # Store the name of the loss function
      self$beta1 = beta1
      self$beta2 = beta2
      self$epsilon = epsilon
      self$patience <- patience
      self$use_early_stopping <- use_early_stopping
      self$regularization <- regularization  # "none", "ridge", "lasso", "elasticnet"


      
      if (loss == "logistique") {
        self$loss_function = self$log_loss
      } else if (loss == "quadratique") {
        self$loss_function = self$mse_loss
      } else {
        stop("Fonction de perte non reconnue")
      }
    },

    
    #' Fit the Multinomial Logistic Regression Model
    #'
    #' This function fits a multinomial logistic regression model to the given data using either the Adam or SGD optimizer.
    #' By default, the model uses early stopping based on the validation loss with a patience of 20 iterations.
    #' It will also use by default the logistique loss function, the Adam optimizer with a mini-batch of 32, and no regularization.
    #' @param X A matrix or data frame of input features.
    #' @param y A factor vector of target labels.
    #' @param validation_split A numeric value indicating the proportion of the data to be used for validation (default is 0.2).
    #'
    #' @details
    #' The function initializes the coefficients, splits the data into training and validation sets, and performs mini-batch gradient descent using the specified optimizer (Adam or SGD). It also includes early stopping based on validation loss.
    #'
    #' The Adam optimizer updates the coefficients using the following formulas:
    #' \deqn{m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t}
    #' \deqn{v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2}
    #' \deqn{\hat{m}_t = \frac{m_t}{1 - \beta_1^t}}
    #' \deqn{\hat{v}_t = \frac{v_t}{1 - \beta_2^t}}
    #' \deqn{\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}
    #'
    #' The SGD optimizer updates the coefficients using the following formula:
    #' \deqn{\theta_t = \theta_{t-1} - \alpha g_t}
    #'
    #' @return None. The function updates the model's coefficients in place.
    #'
    #' @examples
    #' {
    #' model <- LogisticRegressionMultinomial$new()
    #' model$fit(X, y, validation_split = 0.2)
    #' }
    fit = function(X, y, validation_split = 0.2) {
      y <- factor(y)  
      unique_classes <- levels(y)
      num_classes <- length(unique_classes)
      num_samples <- nrow(X)
      num_features <- ncol(X)
      
      # Initialize coefficients (with intercept)
      self$coefficients <- matrix(0, nrow = num_features + 1, ncol = num_classes)
      X <- cbind(1, X)  # Add intercept term
      
      # Split for validation set (early stopping)
      set.seed(123)
      val_indices <- sample(1:num_samples, size = floor(validation_split * num_samples))
      X_val <- X[val_indices, , drop = FALSE]
      y_val <- y[val_indices]
      X_train <- X[-val_indices, , drop = FALSE]
      y_train <- y[-val_indices]

      # Convert to matrix if not the case
      if (!is.matrix(X_train)) {
        X_train <- as.matrix(X_train)
      }
      
      # Adam optimizer variables
      m <- matrix(0, nrow = num_features + 1, ncol = num_classes)
      v <- matrix(0, nrow = num_features + 1, ncol = num_classes)
      best_loss <- Inf
      patience_counter <- 0
      
      for (i in 1:self$num_iterations) {
        # Shuffle for mini-batch
        indices <- sample(1:nrow(X_train))
        X_train <- X_train[indices, , drop = FALSE]
        y_train <- y_train[indices]
        
        # Mini-batch gradient descent
        for (start_idx in seq(1, nrow(X_train), by = self$batch_size)) {
          end_idx <- min(start_idx + self$batch_size - 1, nrow(X_train))
          X_batch <- X_train[start_idx:end_idx, , drop = FALSE]
          y_batch <- y_train[start_idx:end_idx]
          
          # Update coefficients with the selected optimizer
          if (self$optimizer == "adam") {
            result <- self$adam_optimizer(X_batch, y_batch, m, v, self$beta1, self$beta2, self$learning_rate, self$epsilon, i, self$coefficients)
            self$coefficients <- result$coefficients
            m <- result$m
            v <- result$v
          } else if (self$optimizer == "sgd") {
            self$coefficients <- self$sgd_optimizer(X_batch, y_batch, self$learning_rate, self$coefficients)
          } else {
            stop("Invalid optimizer. Choose 'adam' or 'sgd'.")
          }
        }
        
        # Validation set for early stopping
        val_loss <- self$validate(X_val, y_val, unique_classes)
        self$loss_history[i] <- val_loss
        cat("Iteration:", i, "Validation Loss:", val_loss, "\n")
        
        # Early stopping
        if (val_loss < best_loss) {
          best_loss <- val_loss
          patience_counter <- 0
        } else {
          patience_counter <- patience_counter + 1
        }
        
        if (self$use_early_stopping && patience_counter >= self$patience) {
          cat("Early stopping at iteration", i, "with validation loss:", best_loss, "\n")
          break
        }
      }
    },

    #' Adam Optimizer for Multinomial Logistic Regression
    #'
    #' This function performs a single update of the coefficients using the Adam optimization algorithm. It computes the loss, gradients, and updates the coefficients based on the first and second moment estimates.
    #' @details The Adam optimizer updates the coefficients using the following formulas:
    #' \deqn{m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t}
    #' \deqn{v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2}
    #' \deqn{\hat{m}_t = \frac{m_t}{1 - \beta_1^t}}
    #' \deqn{\hat{v}_t = \frac{v_t}{1 - \beta_2^t}}
    #' \deqn{\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}
    #' where \eqn{m_t} and \eqn{v_t} are the first and second moment estimates, \eqn{\beta_1} and \eqn{\beta_2} are the exponential decay rates, \eqn{\alpha} is the learning rate, and \eqn{\epsilon} is a small constant for numerical stability.
    #' @param X_batch A matrix of input features for the current batch.
    #' @param y_batch A factor vector of response variables for the current batch.
    #' @param m A matrix of the first moment estimates.
    #' @param v A matrix of the second moment estimates.
    #' @param beta1 The exponential decay rate for the first moment estimates.
    #' @param beta2 The exponential decay rate for the second moment estimates.
    #' @param learning_rate The learning rate for the optimizer.
    #' @param epsilon A small constant for numerical stability.
    #' @param i The current iteration number.
    #' @param coefficients A matrix of current coefficients.
    #'
    #' @return A list containing the updated coefficients, first moment estimates (m), second moment estimates (v), and the loss value.
    adam_optimizer = function(X_batch, y_batch, m, v, beta1, beta2, learning_rate, epsilon, i, coefficients) {
      # Encode the response variable 
      unique_classes <- levels(y_batch)
      one_hot_y <- self$one_hot_encode(y_batch, unique_classes)
      
      # Loss
      linear_model <- X_batch %*% coefficients
      probabilities <- self$softmax(linear_model)
      loss <- self$log_loss(one_hot_y, probabilities)
      
      error <- probabilities - one_hot_y
      gradient <- t(X_batch) %*% error / nrow(X_batch)
      
      # Update coefficients with Adam optimizer
      m <- beta1 * m + (1 - beta1) * gradient
      v <- beta2 * v + (1 - beta2) * (gradient ^ 2)
      m_hat <- m / (1 - beta1 ^ i)
      v_hat <- v / (1 - beta2 ^ i)
      
      coefficients <- coefficients - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
      
      return(list(coefficients = coefficients, m = m, v = v, loss = loss))
    },

    #' @description This function performs a single step of Stochastic Gradient Descent (SGD) optimization for multinomial logistic regression.
    #' @param X_batch A matrix of input features for the current batch.
    #' @param y_batch A factor vector of target labels for the current batch.
    #' @param learning_rate A numeric value representing the learning rate for SGD.
    #' @param coefficients A matrix of current coefficients for the logistic regression model.
    #' @return A matrix of updated coefficients after performing one step of SGD.
    #' @details
    #' The function first converts the target labels `y_batch` into a one-hot encoded matrix. It then computes the linear model as:
    #' \deqn{Z = X_{batch} \cdot \beta}
    #' where \eqn{X_{batch}} is the input feature matrix and \eqn{\beta} are the coefficients.
    #' 
    #' The probabilities are computed using the softmax function:
    #' \deqn{P = \text{softmax}(Z)}
    #' 
    #' The error is calculated as the difference between the predicted probabilities and the one-hot encoded target labels:
    #' \deqn{\text{error} = P - \text{one\_hot\_y}}
    #' 
    #' The gradient of the loss with respect to the coefficients is computed as:
    #' \deqn{\nabla L = \frac{1}{N} X_{batch}^T \cdot \text{error}}
    #' where \eqn{N} is the number of samples in the batch.
    #' 
    #' Finally, the coefficients are updated using the gradient and the learning rate:
    #' \deqn{\beta = \beta - \text{learning\_rate} \cdot \nabla L}
    #' 
    #' @export
    sgd_optimizer = function(X_batch, y_batch, learning_rate, coefficients) {
      unique_classes <- levels(y_batch)
      one_hot_y <- self$one_hot_encode(y_batch, unique_classes)
      
      # Compute probabilities and gradients
      linear_model <- X_batch %*% coefficients
      probabilities <- self$softmax(linear_model)
      error <- probabilities - one_hot_y
      gradient <- t(X_batch) %*% error / nrow(X_batch)
      
      # Update coefficients with SGD
      coefficients <- coefficients - learning_rate * gradient
      return(coefficients)
    },


    #' @description This function validates the model using the provided validation data.
    #' @param X_val A matrix of validation features.
    #' @param y_val A vector of validation labels.
    #' @param unique_classes A vector of unique class labels.
    #' @return The log loss of the validation data.
    #' @examples
    #' # Assuming `model` is an instance of the logistic regression model
    #' loss <- model$validate(X_val, y_val, unique_classes)
    #' @export
    validate = function(X_val, y_val, unique_classes) {
      val_probabilities <- self$softmax(X_val %*% self$coefficients)
      val_one_hot_y <- self$one_hot_encode(y_val, unique_classes)
      loss <- self$log_loss(val_one_hot_y, val_probabilities)
      return(loss)
    },

    
    #' @description Computes the softmax of the input matrix. The softmax function is used to convert the linear model outputs into class probabilities, projecting them into the range [0, 1].
    #' @details The softmax function is defined as:
    #' \deqn{softmax(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}}
    #' where \(z_i\) is the \(i\)-th element of the input matrix \(z\).
    #' @param z A matrix of linear model outputs.
    #' @return A matrix of softmax probabilities for each class.
    #' @examples
    #' # Assuming `z` is a matrix of linear model outputs
    #' probabilities <- softmax(z)
    #' @export
    softmax = function(z) {
      exp_z = exp(z - apply(z, 1, max))  # Subtract max per row to prevent overflow
      return(exp_z / rowSums(exp_z))
    },
    
    #' @description One-hot encodes the response variable, converting it into a binary matrix. Each row corresponds to a sample, and each column corresponds to a class label.
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

    
    #' @description Predicts the class labels for new data. The function calculates logits and converts them into class probabilities using the softmax function.
    #' It then returns the class with the highest probability for each sample.
    #' @param X A data frame or matrix of predictors, where rows are samples and columns are features.
    #' 
    #' @return A vector of predicted class labels for each sample.
    predict = function(X) {
      X <- cbind(1, X)  # Add intercept term
      linear_model <- X %*% self$coefficients
      probabilities <- self$softmax(linear_model)
      # class_indices <- apply(probabilities, 1, which.max) # Find the class with the highest probability
      # class_labels <- levels(self$y)[class_indices]  # Convert indices to class labels
      # return(class_labels)
      return(apply(probabilities, 1, which.max))  # Convert back to 0 and 1 instead of 1 and 2
    },
    
    #' @description This function calculates the importance of each feature based on the absolute value of the coefficients. It averages the absolute coefficients across all classes and sorts them in descending order.
    #' @return A vector of feature importance scores, sorted in descending order.
    #' @export
    var_importance = function() {
      coef_matrix <- abs(self$coefficients[-1, ])  # Exclure l'intercept
      feature_names <- colnames(self$coefficients)[-1]  # Récupérer les noms des colonnes
      
      # Importance par classe
      importance_scores <- rowMeans(coef_matrix)  # Moyenne des coefficients absolus pour toutes les classes
      importance_ranked <- sort(importance_scores, decreasing = TRUE) # Trier par ordre décroissant
      
      # Afficher les importances
      cat("Variable Importance (sorted):\n")
      for (i in seq_along(importance_ranked)) {
        cat(names(importance_ranked)[i], ": ", round(importance_ranked[i], 4), "\n")
      }
      
      # Retourner les scores
      #return(importance_ranked)
    },


    #' @description This function plots the loss history to visualize the convergence of the loss function over iterations. 
    #' @details The function checks if the loss history is available and non-empty. If the loss history is empty, it stops and prompts the user to run the 'fit' method first. Otherwise, it plots the loss history.
    #' @return A plot showing the convergence of the loss function over iterations.
    #' @examples
    #' {
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

    #' @description This function calculates and plots the ROC AUC for the model predictions. It uses the One vs All strategy to calculate the ROC AUC for each class and plots the ROC curve for each class.
    #' @param X_test A data frame or matrix containing the test features.
    #' @param y_test A vector containing the true labels for the test data.
    #' @param probabilities A matrix of class probabilities for the test data. If not provided, the model will predict probabilities using the test features.
    #' @details The function calculates the ROC AUC for each class using the One vs All strategy. It then plots the ROC curve for each class and displays the AUC value for each class.
    #' @return A plot showing the ROC curve and the AUC value.
    #' @examples
    #' {
    #' model$plot_auc(X_test, y_test)
    #' }
    #' @import pROC
    #' @export
    plot_auc = function(X_test, y_test, probabilities = NULL) {
      library(pROC)
      
      # Predict probabilities if not provided
      if (is.null(probabilities)) {
        X_test <- cbind(1, X_test)  # Add intercept term
        linear_model <- X_test %*% self$coefficients
        probabilities <- self$softmax(linear_model)
      }
      
      # Ensure y_test is a factor
      y_test <- factor(y_test)
      levels_y_test <- levels(y_test)
      
      # Calculate ROC AUC for each class strategy One vs All
      auc_values <- numeric(ncol(probabilities))
      
      # Initialize an empty plot
      plot(NULL, xlim = c(0, 1), ylim = c(0, 1), xlab = "1 - Specificity (False Positive Rate)", 
          ylab = "Sensitivity (True Positive Rate)", main = "ROC Curve", type = "n", asp = 1)
      abline(a = 0, b = 1, lty = 2, col = "gray") # Add diagonal reference line
      
      # Loop through each class
      for (i in 1:ncol(probabilities)) {
        binary_response <- as.numeric(y_test == levels_y_test[i])
        roc_curve <- roc(binary_response, probabilities[, i], quiet = TRUE) # library pROC
        auc_values[i] <- auc(roc_curve)
        
        # Add ROC curve to the plot
        lines(1 - roc_curve$specificities, roc_curve$sensitivities, col = i + 1, lwd = 2)
      }
      
      # Add legend to distinguish between classes
      legend("bottomright", legend = levels_y_test, col = 2:(ncol(probabilities) + 1), lwd = 2)
      
      # Print AUC values
      cat("AUC values for each class:\n")
      for (i in 1:length(auc_values)) {
        cat("Class", levels_y_test[i], ":", auc_values[i], "\n")
      }
    },
    
    #' @description Displays the hyperparameters of the trained model.
    #' @details This function prints out the hyperparameters of the model, including the optimizer, learning rate, number of iterations, and loss function. If the optimizer is "adam", it also prints out the Adam-specific parameters: Beta1, Beta2, and Epsilon.
    #' It will also print out the regularization method, batch size, and early stopping parameters.
    #' @return None. This function is used for its side effect of printing the model's hyperparameters.
    #' @examples
    #' {
    #' model$summary()
    #' }
    summary = function() {
      # Model Hyperparameters
      cat("\n=== Model Hyperparameters ===\n")
      cat("Optimizer: ", self$optimizer, "\n")
      cat("Learning Rate: ", self$learning_rate, "\n")
      cat("Number of Iterations: ", self$num_iterations, "\n")
      cat("Loss Function: ", self$loss_name, "\n")
      cat("Early Stopping: ", ifelse(self$use_early_stopping, "Enabled", "Disabled"), "\n")
      cat("Patience: ", self$patience, "\n")
      cat("Regularization: ", self$regularization, "\n")
      cat("Batch Size: ", self$batch_size, "\n")
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
    #' {
    #' model$print(X_test, y_test)
    #' }
    #'
    #' @import caret
    #' @import MLmetrics
    #' @export
    print = function(X_test, y_test) {
      cat("\n=== Model Metrics ===\n")

      probabilities <- self$predict_proba(X_test)
      predictions <- self$predict(X_test)
      
      # # Confusion Matrix # Already included in the caret report
      # confusion_matrix <- table(Predicted = predictions, Actual = y_test)
      # print("Confusion Matrix:")
      # print(confusion_matrix)
      

      #  F1-score, precision, Recall, AUC
      library(caret)
      library(MLmetrics)
      report <- confusionMatrix(as.factor(predictions), as.factor(y_test))

      print(report)
      
      f1_weighted <- F1_Score(y_pred = predictions, y_true = y_test) # use MLmetrics
      cat("F1 Score:", f1_weighted, "\n")

      self$plot_auc(X_test, y_test, probabilities)
      
        output <- capture.output({
        print("Confusion Matrix:")
        # print(confusion_matrix)
        print(report)
        cat("F1 Score:", f1_weighted, "\n")
      })
      
      return(output)
    },
    
      
      

    #' @description This function computes the log loss, also known as logistic loss or cross-entropy loss, 
    #' between the true labels and the predicted probabilities. The log loss is calculated as:
    #' \deqn{-\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]}
    #' where \eqn{N} is the number of samples, \eqn{y_i} is the true label of the i-th sample, and \eqn{p_i} is the predicted probability of the i-th sample.
    #' @param y_true A numeric vector of true labels.
    #' @param y_pred A numeric vector of predicted probabilities.
    #' @return A numeric value representing the log loss.
    #' @examples
    #' y_true <- c(1, 0, 1, 0)
    #' y_pred <- c(0.9, 0.1, 0.8, 0.2)
    #' log_loss(y_true, y_pred)
    #' @export
    log_loss = function(y_true, y_pred) {
      epsilon <- 1e-15  # Small value to prevent log(0)
      y_pred <- pmax(pmin(y_pred, 1 - epsilon), epsilon) 
      loss <- -sum(y_true * log(y_pred))  # Régularisez par 1/N ?

      # Add penalty
      reg_results <- self$apply_regularization(NULL, self$coefficients)
      loss <- loss + reg_results$penalty
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
    mse_loss = function(y_true, y_pred) {
      0.5 * mean((y_true - y_pred)^2)
    },

    # mse_loss = function(y_true, y_pred) { # MSE MULTINOMIALE?
    #   0.5 * sum((y_true - y_pred)^2)
    # }

    #' @description Predicts the class probabilities for new data.
    #' @param X A data frame or matrix of predictors, where rows are samples and columns are features.
    #' @return A matrix of predicted class probabilities for each sample.
    predict_proba = function(X) {
      X <- cbind(1, X)  # Add intercept term
      linear_model <- X %*% self$coefficients
      probabilities <- self$softmax(linear_model)
      return(probabilities)
    },


    #' @description Displays the selected variables based on their importance scores.
    #' @param num_variables An integer specifying the number of top variables to display.
    #' @details This function calculates the importance of each feature based on the absolute value of the coefficients. It sums the absolute coefficients for each feature and selects the top 'num_variables' features based on their importance scores.
    #' @return None. This function is used for its side effect of printing the selected variables.
    #' @examples
    #' {
    #' model$select_variables(num_variables = 5)
    #' }
    select_variables = function(num_variables) {
      # Calculate the importance of each feature based on the absolute value of the coefficients
      coef_matrix <- abs(self$coefficients[-1, ])  # Exclude the intercept term
      importance_scores <- rowSums(coef_matrix)    # Sum of absolute coefficients for each feature
      importance_ranked <- sort(importance_scores, decreasing = TRUE)
      
      # Select the top 'num_variables' features
      top_variables <- names(importance_ranked)[1:num_variables]
      
      # Print the selected variables
      cat("Selected Variables:\n")
      for (i in 1:length(top_variables)) {
        cat(top_variables[i], "\n")
      }
    },

    #' @description Applies regularization to the gradient and computes the penalty term for the loss function.
    #' @param gradient A matrix of gradients with respect to the model coefficients.
    #' @param coefficients A matrix of model coefficients, where the first row corresponds to the intercept.
    #' @param p A numeric value (default 0.5) representing the mixing parameter for ElasticNet regularization.
    #' @return A list containing:
    #'   - `penalty`: The computed penalty term to be added to the loss function.
    #'   - `regularized_gradient`: The gradient matrix adjusted for regularization.
    #' @details The ElasticNet regularization combines L1 and L2 penalties. The penalty term is computed as:
    #' \deqn{penalty = \frac{\lambda}{2} \left( (1 - p) \sum_{j=1}^{n} \beta_j^2 + p \sum_{j=1}^{n} |\beta_j| \right)}
    #' where \(\lambda\) is the regularization parameter, \(p\) is the mixing parameter, \(\beta_j\) are the model coefficients, and \(n\) is the number of coefficients.
    #' 
    #' For Ridge regularization, the penalty term is computed as:
    #' \deqn{penalty = \frac{1}{2} \sum_{j=1}^{n} \beta_j^2}
    #' where \(\beta_j\) are the model coefficients.
    #' 
    #' For Lasso regularization, the penalty term is computed as:
    #' \deqn{penalty = \frac{1}{2} \sum_{j=1}^{n} |\beta_j|}
    #' where \(\beta_j\) are the model coefficients.
    apply_regularization = function(gradient, coefficients, p = 0.5) {
      penalty <- 0 
      regularized_gradient <- gradient 

      coef_no_intercept <- coefficients[-1, ]
      
      if (self$regularization == "ridge") {
        # Ridge : 1/2 * sum(beta^2)
        penalty <- 0.5 * sum(coef_no_intercept^2)
        regularized_gradient[-1, ] <- gradient[-1, ] + coef_no_intercept
      } else if (self$regularization == "lasso") {
        # Lasso : 1/2 * sum(|beta|)
        penalty <- 0.5 * sum(abs(coef_no_intercept))
        regularized_gradient[-1, ] <- gradient[-1, ] + sign(coef_no_intercept)
      } else if (self$regularization == "elasticnet") {
        # ElasticNet : (1-p)/2 * sum(beta^2) + p * sum(|beta|)
        penalty <- 0.5 * (1 - p) * sum(coef_no_intercept^2) + 0.5 * p * sum(abs(coef_no_intercept))
        regularized_gradient[-1, ] <- gradient[-1, ] + (1 - p) * coef_no_intercept + p * sign(coef_no_intercept)
      }
      
      return(list(penalty = penalty, regularized_gradient = regularized_gradient))
    },
    
    
    #' @description Exports the trained model to a PMML (Predictive Model Markup Language) file.
    #' @param file_path A string specifying the path where the PMML file will be saved.
    #' @return Saves the PMML representation of the trained model to the specified file and returns a success message.
    #' @details This function generates a PMML file for a multinomial logistic regression model, including the model's
    #'   coefficients and metadata. It ensures that the model is trained before exporting and uses the PMML version 4.4 format.
    export_pmml = function(file_path) {
      # Check if the model is trained
      if (is.null(self$coefficients)) {
        stop("Error: model must be trained before being exported.")
      }
      
      # XML library
      library(XML)
      
      # Root node for PMML
      pmml <- newXMLNode("PMML", namespaceDefinitions = c("http://www.dmg.org/PMML-4_4"), attrs = c(version = "4.4"))
      
      # Add header
      header <- newXMLNode("Header", parent = pmml)
      newXMLNode("Application", attrs = c(name = "LogisticRegressionMultinomial", version = "1.0"), parent = header)
      newXMLNode("Timestamp", Sys.time(), parent = header)
      
      # Add data dictionary
      model <- newXMLNode("RegressionModel", attrs = c(functionName = "classification", algorithmName = "multinomial logistic regression"), parent = pmml)
      newXMLNode("MiningSchema", newXMLNode("MiningField", attrs = c(name = "target", usageType = "target")), parent = model)
      
      # Add coefficients of the model
      regression_table <- newXMLNode("RegressionTable", parent = model)
      for (class_index in seq_len(ncol(self$coefficients))) {
        for (feature_index in seq_len(nrow(self$coefficients))) {
          coefficient <- self$coefficients[feature_index, class_index]
          newXMLNode("NumericPredictor", attrs = c(name = paste0("Feature", feature_index), coefficient = coefficient), parent = regression_table)
        }
      }
      
      # Save PMML to file
      saveXML(pmml, file = file_path)
      message("Model exported successfully. ", file_path)
    }
   
  )
)


# nolint end