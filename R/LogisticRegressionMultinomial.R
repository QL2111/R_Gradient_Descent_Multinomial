
# nolint start
# Générer la documentation
# roxygen2::roxygenise()

# 14/11 -> Le test sur credit_card_rmd a montré que le problème vient du modèle et non du préprocessing #### OK

#' @TODO: 
#' Rshiny -> Utiliser une librairie, retaper
#' Pouvoir choisir plusieurs régularisations (L1, L2, ElasticNet) # Daniella # EN COURS
#' #' # Implémenter analyse factorielle dans le datapreparer + tester avec studentperformance # Quentin   #### OK
#' #' Incorporer AFDM dans data preparer # Quentin  ncp pour le nombre de dimensions à garder(variables explicatives cumulé>95%) # Quentin #### OK MAIS accuracy faible pour student performance
#' Ajouter var select # Awa #### à tester - Quentin (select_variables)
#' Changer les levels ? Répréesentation en 1,2,3 mais plus tard garder les labels? # Quentin # Casse les autres fonctions -> Laisser pour l'isntatn
#' Mettre un Imputer sur le datapreparer, Missing values aussi à mettre dans le datapreparer et outliers avant le scaler # Quentin
#' ReadMe Github 
#' Video explicative(tuto)
#' legends (nom des classes) auc PLOT # Quentin (à voir si on garde ? Rshiny)
#' Améliroer le roc AUC dans shiny(éviter de calculer 2 fois) # Quentin
#' Formulaire Shiny, rajouter l'option d'analyse factorielle et de régularisation # Daniella
#' Device model mauvais test -> essayer avec une autre variable cible(User Behavior classification pour voir si l'accuracy monte) # Awa
#' help # Daniella/Quentin
#' Ajouter régularisation + export PMML dans LogisticRegressionMultinomial dans LogistRegression.R # Quentin
#' SMOTE # Quentin
#' Imputation par KNN ? # Quentin -> Inclure dans le rapport discussion, jeu de données lourd
#' Outliers ? #Quentin
#' @PACKAGE IMPORTER
#' Peut-être ne pas utiliser caret() + MLmetrics + pROC +  stats(mode) + pml
#' @NEXT
#' 
#' #' revoir SGD
#' #' FIT REGRESSION LOGISTIQUE VOIR STRATEGIE Mini Batch(nb paramètre de l'algorithme) au lieu de Batch Gradient Descent(Tout l'ensemble de données) 
#' @BONUS
#' Mettre en image Docker
#' #' Paralleliser les calculs
#' #' Analyse Factorielle (Plus de dimension) # Awa
#' #' R Shiny -> Ajouter nouveaux champ pour les hyperparamètres du modèles,  #### EN COURS + de champs possibles ?

#' 
#' @DONE
#' #' Exportation en PMML # Daniella ### OK
#' #' Tester Analyse factorielle multiclass tester avec student_performancce + Iris + JEU DE DONNEES avec beaucoup de col # Awa Iris + StudentPerformance # OK
#' #' intégrer le train/test split dans le datapreparer  + stratify # Quentin ### OK
#' #' INCORPORER D'autres métriques(print) (F1, precision, recall, ROC AUC, etc.  probabilité d'appartenance aux classes) # Daniella
#' #' AUC ? -> print + shiny # Quentin ####ok
#' #' Pouvoir choisir plusieurs optimiseurs (Adam, SGD, etc.) # Awa(fit) #### LaTeX SGD pas efficace ?
#' Tester var_importance et comparer avec sklearn # Quentin         #### OK
#' #' predict_proba() pour avoir les probabilités des classes + ajouter au summary # Daniella # A REVOIR DT sklearn
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
#' #' Tester avec StudentPerformance # Daniella Quentin OK #### A REVOIR
#' mini batch au lieu de online ? (GRadient descent)



#' #' Exportation sous forme de package R # Quentin  
#' #### OK devtools::build() 
#' Pour l'installer
#' install.packages("mon_package_0.1.0.tar.gz", repos = NULL, type = "source") 
#' installer avec github
#' devtools::install_github("Lien du repo")



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
LogisticRegressionMultinomial = R6Class("LogisticRegressionMultinomial",
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

    regularization = NULL, #type (none, l1, l2, elasticnet)
    lambda = NULL, #coefficient
    alpha = NULL, #for elasticnet
    
    # class_labels = NULL,  # Store the class labels to rename them later
    
    #' @description Initializes a new instance of the `LogisticRegressionMultinomial` class.
    #' @param learning_rate Numeric. Sets the learning rate for gradient descent. Default is 0.01.
    #' @param num_iterations Integer. Specifies the number of gradient descent iterations. Default is 1000.
    #' @param loss Character. Specifies the loss function to use. Options are "logistique", "quadratique", "deviance". Default is "logistique".
    #' @param optimizer Character. Specifies the optimizer to use. Options are "adam", "sgd". Default is "adam".
    #' @param use_early_stopping Logical. Whether to use early stopping. Default is TRUE.
    #' @param patience Integer. Number of iterations to wait for improvement before stopping early. Default is 10.
    #' @return A new `LogisticRegressionMultinomial` object.
    initialize = function(learning_rate = 0.01, num_iterations = 1000, loss = "logistique", optimizer = "adam", beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, patience = 20,
     use_early_stopping = TRUE, regularization = "l2", lambda = 1, alpha = 0.5) {
      self$learning_rate = learning_rate
      self$num_iterations = num_iterations
      self$loss_history = numeric(num_iterations)
      self$optimizer = optimizer
      self$loss_name = loss  # Store the name of the loss function
      self$beta1 = beta1
      self$beta2 = beta2
      self$epsilon = epsilon
      self$patience = patience
      self$use_early_stopping = use_early_stopping
      self$regularization = regularization
      self$lambda = lambda
      self$alpha = alpha


      
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
    #' @param validation_split Numeric. Fraction of the training data to be used as validation data. Default is 0.2.
    #' @details The `fit` method initializes model coefficients and applies gradient descent to minimize the loss function. It calculates class probabilities with softmax and updates coefficients based on the gradient.
    #' @return No return value; updates the model's coefficients.
    fit = function(X, y, validation_split = 0.2) {
      y = factor(y)  # Convert y to factor to ensure consistent class levels

      # self$class_labels = levels(y)  # Store the class labels for later use

      unique_classes = levels(y)  # Use levels of factor y      num_classes = length(unique_classes)
      num_samples = nrow(X)
      num_features = ncol(X)
      num_classes = length(unique_classes)

      if (self$use_early_stopping) {
        # Split data into training and validation sets
        set.seed(123)  # For reproducibility
        validation_indices = sample(1:num_samples, size = floor(validation_split * num_samples))
        X_val = X[validation_indices, ]
        y_val = y[validation_indices]
        X_train = X[-validation_indices, ]
        y_train = y[-validation_indices]
      } else {
        X_train = X
        y_train = y
        X_val = NULL
        y_val = NULL
      }
      
      # Initialize coefficients
      self$coefficients = matrix(0, nrow = num_features + 1, ncol = num_classes)
      X_train = cbind(1, X_train)  # Add intercept term
      if (!is.null(X_val)) {
        X_val = cbind(1, X_val)  # Add intercept term
      }
      
      best_loss = Inf
      patience_counter = 0
      
      if (self$optimizer == "adam") {
        self$adam_optimizer(X_train, y_train, X_val, y_val, unique_classes, num_samples, num_features, num_classes, best_loss, patience_counter)
      } else if (self$optimizer == "sgd") {
        self$sgd_optimizer(X_train, y_train, X_val, y_val, unique_classes, num_samples, num_features, num_classes, best_loss, patience_counter)
      } else {
        stop("Optimiseur non reconnu")
      }
    },

    
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
        one_hot[i, as.integer(y[i])] = 1
      }
      return(one_hot)
    },

    
    #' @description Predicts the class labels for new data.
    #' @param X A data frame or matrix of predictors, where rows are samples and columns are features.
    #' @return A vector of predicted class labels for each sample.
    predict = function(X) {
      X = cbind(1, X)  # Add intercept term
      linear_model = X %*% self$coefficients
      probabilities = self$softmax(linear_model)
      # class_indices = apply(probabilities, 1, which.max) # Find the class with the highest probability
      # class_labels = levels(self$y)[class_indices]  # Convert indices to class labels
      # return(class_labels)
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
      coef_matrix = abs(self$coefficients[-1, ])  # Exclure l'intercept
      feature_names = colnames(self$coefficients)[-1]  # Récupérer les noms des colonnes
      
      # Importance par classe
      importance_scores = rowMeans(coef_matrix)  # Moyenne des coefficients absolus pour toutes les classes
      importance_ranked = sort(importance_scores, decreasing = TRUE) # Trier par ordre décroissant
      
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

    #' @description This function calculates and plots the ROC AUC for the model predictions.
    #' @param X_test A data frame or matrix containing the test features.
    #' @param y_test A vector containing the true labels for the test data.
    #' @return A plot showing the ROC curve and the AUC value.
    #' @examples
    #' \dontrun{
    #' model$plot_auc(X_test, y_test)
    #' }
    #' @import pROC
    #' @export
    plot_auc = function(X_test, y_test, probabilities = NULL) {
      library(pROC)
      
      # Predict probabilities if not provided
      if (is.null(probabilities)) {
        X_test = cbind(1, X_test)  # Add intercept term
        linear_model = X_test %*% self$coefficients
        probabilities = self$softmax(linear_model)
      }
      
      # Ensure y_test is a factor
      y_test = factor(y_test)
      levels_y_test = levels(y_test)
      
      # Calculate ROC AUC for each class strategy One vs All
      auc_values = numeric(ncol(probabilities))
      
      # Initialize an empty plot
      plot(NULL, xlim = c(0, 1), ylim = c(0, 1), xlab = "1 - Specificity (False Positive Rate)", 
          ylab = "Sensitivity (True Positive Rate)", main = "ROC Curve", type = "n", asp = 1)
      abline(a = 0, b = 1, lty = 2, col = "gray") # Add diagonal reference line
      
      # Loop through each class
      for (i in 1:ncol(probabilities)) {
        binary_response = as.numeric(y_test == levels_y_test[i])
        roc_curve = roc(binary_response, probabilities[, i], quiet = TRUE) # library pROC
        auc_values[i] = auc(roc_curve)
        
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
      cat("Regularization: ", self$regularization, "\n")

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
      probabilities = self$predict_proba(X_test)
      predictions = self$predict(X_test)
      
      # # Confusion Matrix # Already included in the caret report
      # confusion_matrix = table(Predicted = predictions, Actual = y_test)
      # print("Confusion Matrix:")
      # print(confusion_matrix)
      

      #  F1-score, precision, Recall, AUC
      library(caret)
      library(MLmetrics)
      report = confusionMatrix(as.factor(predictions), as.factor(y_test))

      print(report)
      
      f1_weighted = F1_Score(y_pred = predictions, y_true = y_test) # use MLmetrics
      cat("F1 Score:", f1_weighted, "\n")

      self$plot_auc(X_test, y_test, probabilities)
      
        output = capture.output({
        print("Confusion Matrix:")
        # print(confusion_matrix)
        print(report)
        cat("F1 Score:", f1_weighted, "\n")
      })
      
      return(output)
    },
    
      
      

    #' @description This function computes the log loss, also known as logistic loss or cross-entropy loss, 
    #' between the true labels and the predicted probabilities.
    #'
    #' @param y_true A numeric vector of true labels.
    #' @param y_pred A numeric vector of predicted probabilities.
    #' @return A numeric value representing the log loss.
    #' @examples
    #' y_true = c(1, 0, 1, 0)
    #' y_pred = c(0.9, 0.1, 0.8, 0.2)
    #' log_loss(y_true, y_pred)
    #' @export
    # LOSS FUNCTIONS
    log_loss = function(y_true, y_pred) {
      epsilon = 1e-15  # Small value to prevent log(0)
      y_pred = pmax(pmin(y_pred, 1 - epsilon), epsilon) 
      loss = -sum(y_true * log(y_pred))  # Régularisez par 1/N ?
      return(loss)

    },

    
    #' @description This function computes the mean squared error (MSE) loss between the true labels 
    #' and the predicted values.
    #'
    #' @param y_true A numeric vector of true labels.
    #' @param y_pred A numeric vector of predicted values.
    #' @return A numeric value representing the MSE loss.
    #' @examples
    #' y_true = c(1, 0, 1, 0)
    #' y_pred = c(0.9, 0.1, 0.8, 0.2)
    #' mse_loss(y_true, y_pred)
    #' @export
    #'
    mse_loss = function(y_true, y_pred) {
      0.5 * mean((y_true - y_pred)^2)
    },

    # mse_loss = function(y_true, y_pred) { # MSE MULTINOMIALE?
    #   0.5 * sum((y_true - y_pred)^2)
    # }

    apply_regularization = function(gradient, regularization, lambda, alpha) {
      if (self$regularization == "l2") {
        # Ridge: L2 (penalty is lambda * coefficients)
        return(gradient + self$lambda * self$coefficients)
      } else if (self$regularization == "l1") {
        # Lasso: L1 (penalty is lambda * sign(coefficients))
        return(gradient + self$lambda * sign(self$coefficients))
      } else if (self$regularization == "elasticnet") {
        # ElasticNet
        l1_part <- self$alpha * self$lambda * sign(self$coefficients)
        l2_part <- (1 - self$alpha) * self$lambda * self$coefficients
        return(gradient + l1_part + l2_part)
      } else {
        # No regularization
        return(gradient)
      }
    },



    #' @description Adam optimizer for updating coefficients.
    #' @param X Matrix of predictors with intercept term added.
    #' @param y Factor vector of response variable.
    #' @param unique_classes Vector of unique class labels.
    #' @param num_samples Number of samples.
    #' @param num_features Number of features.
    #' @param num_classes Number of classes.
    adam_optimizer = function(X_train, y_train, X_val, y_val, unique_classes, num_samples, num_features, num_classes, best_loss, patience_counter) {
      m = matrix(0, nrow = num_features + 1, ncol = num_classes)
      v = matrix(0, nrow = num_features + 1, ncol = num_classes)
      one_hot_y = self$one_hot_encode(y_train, unique_classes) 

      for (i in 1:self$num_iterations) {
        linear_model = X_train %*% self$coefficients 
        probabilities = self$softmax(linear_model)
        # one_hot_y = self$one_hot_encode(y_train, unique_classes)
        loss = self$loss_function(one_hot_y, probabilities)
        self$loss_history[i] = loss
        
        cat("Iteration:", i, "Loss:", loss, "\n")

        error = probabilities - one_hot_y
        gradient = t(X_train) %*% error / num_samples

        # Apply regularization
        if (!is.null(self$regularization)) {
          gradient <- gradient + self$apply_regularization(self$coefficients, self$regularization, lambda, alpha)
        }

        m = self$beta1 * m + (1 - self$beta1) * gradient
        v = self$beta2 * v + (1 - self$beta2) * (gradient ^ 2)
        
        m_hat = m / (1 - self$beta1 ^ i)
        v_hat = v / (1 - self$beta2 ^ i)

        self$coefficients = self$coefficients - self$learning_rate * m_hat / (sqrt(v_hat) + self$epsilon)

        # Early stopping
        if (self$use_early_stopping) {
          val_probabilities = self$softmax(X_val %*% self$coefficients)
          val_one_hot_y = self$one_hot_encode(y_val, unique_classes)
          val_loss = self$loss_function(val_one_hot_y, val_probabilities)
          
          if (val_loss < best_loss) {
            best_loss = val_loss
            patience_counter = 0
          } else {
            patience_counter = patience_counter + 1
          }
          
          if (patience_counter >= self$patience) {
            cat("Early stopping at iteration:", i, "with validation loss:", val_loss, "\n")
            break
          }
        }
      }
    },

    #' @description SGD optimizer for updating coefficients.
    #' @param X Matrix of predictors with intercept term added.
    #' @param y Factor vector of response variable.
    #' @param unique_classes Vector of unique class labels.
    #' @param num_samples Number of samples.
    #' @param num_features Number of features.
    #' @param num_classes Number of classes.
    sgd_optimizer = function(X_train, y_train, X_val, y_val, unique_classes, num_samples, num_features, num_classes, best_loss, patience_counter) {
      for (i in 1:self$num_iterations) {
        linear_model = X_train %*% self$coefficients
        probabilities = self$softmax(linear_model)
        one_hot_y = self$one_hot_encode(y_train, unique_classes)
        loss = self$loss_function(one_hot_y, probabilities)
        self$loss_history[i] = loss
        
        cat("Iteration:", i, "Loss:", loss, "\n")

        error = probabilities - one_hot_y
        gradient = t(X_train) %*% error / num_samples

        # Apply regularization
        if (!is.null(self$regularization)) {
          gradient <- gradient + self$apply_regularization(self$coefficients, self$regularization, lambda, alpha)
        }


        self$coefficients = self$coefficients - self$learning_rate * gradient

        # Early stopping
        if (self$use_early_stopping) {
          val_probabilities = self$softmax(X_val %*% self$coefficients)
          val_one_hot_y = self$one_hot_encode(y_val, unique_classes)
          val_loss = self$loss_function(val_one_hot_y, val_probabilities)
          
          if (val_loss < best_loss) {
            best_loss = val_loss
            patience_counter = 0
          } else {
            patience_counter = patience_counter + 1
          }
          
          if (patience_counter >= self$patience) {
            cat("Early stopping at iteration:", i, "with validation loss:", val_loss, "\n")
            break
          }
        }
      }
    },


    #' @description Predicts the class probabilities for new data.
    #' @param X A data frame or matrix of predictors, where rows are samples and columns are features.
    #' @return A matrix of predicted class probabilities for each sample.
    predict_proba = function(X) {
      X = cbind(1, X)  # Add intercept term
      linear_model = X %*% self$coefficients
      probabilities = self$softmax(linear_model)
      return(probabilities)
    },

    export_pmml = function(file_name = "model.pmml") {
      if (is.null(self$coefficients)) {
        stop("Model not yet fitted. Please run fit() first.")
      }
      
      # Construction manuelle d'un objet modèle compatible
      pmml_model <- pmml::pmml(
        model = self,  # Vous devez adapter cet objet pour correspondre aux besoins de pmml
        model.name = "LogisticRegressionMultinomial",
        app.name = "Student Performance Logistic Model",
        description = "A custom logistic regression model for Access_to_Resources",
        coefficients = self$coefficients,
        targetField = "Access_to_Resources",  # La variable cible
        inputFields = colnames(X_train),  # Les noms des prédicteurs
        fieldDescription = NULL  # Description des champs (facultatif)
      )
      
      # Exportation
      pmml::writePMML(pmml_model, file_name)
      cat("Model exported to PMML file:", file_name, "\n")
    },
    


    #' Select Important Variables Based on Coefficients
    #'
    #' This function selects the most important variables based on the absolute value of the coefficients
    #' from a logistic regression model. It calculates the importance of each feature, ranks them, and 
    #' selects the top `num_variables` features.
    #'
    #' @param num_variables An integer specifying the number of top variables to select.
    #' @return A character vector containing the names of the selected top variables.
    #' @examples
    #' \dontrun{
    #'   selected_vars = select_variables(5)
    #'   print(selected_vars)
    #' }
    #' @export
    select_variables = function(num_variables) {
      # Calculate the importance of each feature based on the absolute value of the coefficients
      coef_matrix = abs(self$coefficients[-1, ])  # Exclude the intercept term
      importance_scores = rowSums(coef_matrix)    # Sum of absolute coefficients for each feature
      importance_ranked = sort(importance_scores, decreasing = TRUE)
      
      # Select the top 'num_variables' features
      top_variables = names(importance_ranked)[1:num_variables]
      
      # Print the selected variables
      cat("Selected Variables:\n")
      for (i in 1:length(top_variables)) {
        cat(top_variables[i], "\n")
      }
      
      # Return the selected features as a subset of the original data
      return(top_variables)
    }

    
  )
)


# nolint end
