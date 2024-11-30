# nolint start
# DataPreparer.R
# Générer doc
# devtools::document("D:/GitHub/R_Gradient_Descent_Multinomial")
# LIBRARY CARET createDataPartition, librarie factoextra et FactoMineR pour l'analyse factorielle
# Préprocess la variable cible peut poser des problèmes ? Cas de variable cible encodée en numérique, il ne faudrait pas la standardiser


#' @title Data Preparation Class
#' @description The `DataPreparer` class provides methods to standardize quantitative data and encode qualitative data, allowing the option of factor analysis for mixed data.
#' @details This class is part of a package designed to support data preparation tasks, particularly for models that need standardized quantitative features and encoded categorical features. The class can handle mixed data types and offers both one-hot encoding and an alternative factor analysis encoding for qualitative data.
#' 
#' @field use_factor_analysis A logical flag indicating whether to apply factor analysis on qualitative data instead of one-hot encoding. Default is \code{FALSE}.
#' 
#' @export
DataPreparer = R6::R6Class("DataPreparer", 
  public = list(
    #' @field use_factor_analysis Logical. Indicates whether to apply factor analysis for qualitative data encoding.
    use_factor_analysis = FALSE,
    
    #' @description Initializes a new instance of the `DataPreparer` class.
    #' @param use_factor_analysis Logical. Set to `TRUE` to enable factor analysis for qualitative variables, `FALSE` for one-hot encoding.
    #' @return A new `DataPreparer` object.
    initialize = function(use_factor_analysis = FALSE) {
      if (!is.logical(use_factor_analysis)) {
        stop("use_factor_analysis must be a logical value (TRUE or FALSE)")
      }
      self$use_factor_analysis <- use_factor_analysis
    },

    split_data = function(data, target_col, split_ratio = 0.7, stratify = FALSE) {
      library(caret)
      
      set.seed(123)  # Pour la reproductibilité
      
      if (stratify) {
        # Split stratifié
        train_indices = createDataPartition(data[[target_col]], p = split_ratio, list = FALSE)
      } else {
        # Split non stratifié
        train_indices = sample(seq_len(nrow(data)), size = floor(split_ratio * nrow(data)))
      }
      
      train_data = data[train_indices, ]
      test_data = data[-train_indices, ]
      
      return(list(train = train_data, test = test_data))
    },

    prepare_data = function(data, target_col, split_ratio = 0.7, stratify = FALSE, remove_outliers = FALSE, outlier_seuil = 0.25) {

      # Remoove outliers avant standardisation et imputation
      if (remove_outliers) {
        data = self$remove_outliers(data, seuil = outlier_seuil)
      }

      # Imputation by mode/median
      data = self$handle_missing_data(data)

      # Split the data
      split = self$split_data(data, target_col, split_ratio, stratify)
      train_data = split$train
      test_data = split$test
      
      # Process training data
      train_data_prepared = self$process_data(train_data)
      
      # Process test data
      test_data_prepared = self$process_data(test_data)
      
      # Extract features and target
      X_train = train_data_prepared[, colnames(train_data_prepared) != target_col]
      y_train = train_data[[target_col]]
      X_test = test_data_prepared[, colnames(test_data_prepared) != target_col]
      y_test = test_data[[target_col]]
      
      return(list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test))
    },

    handle_missing_data = function(data) {
      quantitative_vars = sapply(data, is.numeric)
      qualitative_vars = sapply(data, is.factor)
      
      # Imputation pour les variables quantitatives par la médiane
      for (col in names(data)[quantitative_vars]) {
        data[[col]][is.na(data[[col]])] = median(data[[col]], na.rm = TRUE)
      }
      
      # Imputation pour les variables qualitatives par la mode
      for (col in names(data)[qualitative_vars]) {
        mode_value = calculate_mode(data[[col]])
        
        # Ajouter la valeur de la mode aux niveaux du facteur si elle n'existe pas déjà
        if (!mode_value %in% levels(data[[col]])) {
          levels(data[[col]]) = c(levels(data[[col]]), mode_value)
        }
        
        data[[col]][is.na(data[[col]])] = mode_value
      }
      
      return(data)
    },

    
    #' @description Standardizes a quantitative variable.
    #' @param data Numeric vector representing a quantitative variable to be standardized.
    #' @return A standardized numeric vector with mean 0 and standard deviation 1.
    standardize = function(data) {
      return((data - mean(data, na.rm = TRUE)) / sd(data, na.rm = TRUE))
    },
    
    #' @description Prepares data by standardizing quantitative variables and encoding qualitative variables.
    #' @param data A data frame containing both quantitative and qualitative variables.
    #' @return A prepared data frame with standardized quantitative variables and encoded qualitative variables.
    #' @details The `prepare_data` method processes quantitative and qualitative data separately:
    #' \itemize{
    #'   \item \strong{Quantitative data}: Standardizes each quantitative variable to have mean 0 and standard deviation 1.
    #'   \item \strong{Qualitative data}: If `use_factor_analysis` is `TRUE`, applies factor analysis. If `FALSE`, performs one-hot encoding.
    #' }
    process_data = function(data) {
      quantitative_vars = sapply(data, is.numeric) # Check the type isnumeric
      qualitative_vars = !quantitative_vars # Check the type is not numeric
      prepared_list = list()
      
      # Process quantitative variables: standardization
      if (any(quantitative_vars)) {
        # quant_data = data[, quantitative_vars, drop = FALSE] # drop=FALSE to keep data frame or else it will transform into a vector
        # quant_data = as.data.frame(lapply(quant_data, self$standardize)) # Apply the standardize function -> Old version, using scale works better
        # prepared_list$quantitative = quant_data
        quant_data = data[, quantitative_vars, drop = FALSE] # drop=FALSE to keep data frame or else it will transform into a vector
        quant_data = scale(data[, quantitative_vars, drop = FALSE])
        prepared_list$quantitative = quant_data
      }
      
      # Process qualitative variables: one-hot encoding
      if (any(qualitative_vars)) {
        qual_data = data[, qualitative_vars, drop = FALSE]
        # One-hot encoding
        qual_data = as.data.frame(model.matrix(~ . - 1, data = qual_data)) # ~ . - 1 means all columns except the first one (exclude intercept - to avoid multicollinearity)
        prepared_list$qualitative = qual_data
      }
      
      # Combine quantitative and qualitative data
      prepared_data = do.call(cbind, prepared_list)
      
      if (self$use_factor_analysis) {
        library(FactoMineR)
        library(factoextra)
        
        cat("Utilisation de l'Analyse Factorielle des Données Mixtes (AFDM)\n")
        # cat("Dimensions des données envoyées à FAMD :", dim(prepared_data), "\n")

        # Effectuer l'AFDM
        famd_result = FAMD(data, graph = FALSE, ncp = ncol(prepared_data))
        
        # Afficher les dimensions des données pour vérification
        print(dim(prepared_data))

        # Extraire les valeurs propres et calculer la variance expliquée cumulée
        eig_values = famd_result$eig[, 3]  # Colonne de la variance expliquée cumulée (en pourcentage)

        # Trouver le nombre de dimensions nécessaires pour atteindre au moins 90% de la variance expliquée cumulée
        ncp_index = which(eig_values >= 95)[1]
        
        # Si aucune valeur ne dépasse 90%, garder toutes les dimensions
        if (is.na(ncp_index)) {
          ncp = min(ncol(prepared_data), nrow(prepared_data))  # On ne peut pas garder plus que le minimum de lignes ou colonnes
        } else {
          ncp = ncp_index
        }

        cat("Nombre de dimensions retenues :", ncp, "\n")
        
        # Réduire les données aux dimensions nécessaires
        reduced_data = famd_result$ind$coord[, 1:ncp, drop = FALSE]
        return(reduced_data)
      } else {
        cat("Utilisation de l'encodage one-hot \n")
        return(prepared_data)
      }
    },
    
    detect_outliers = function(x, seuil = 0.25) {
      if (is.numeric(x)) {
        Q1 <- quantile(x, seuil, na.rm = TRUE)
        Q3 <- quantile(x, 1 - seuil, na.rm = TRUE)
        IQR <- Q3 - Q1
        lower_bound <- Q1 - 1.5 * IQR
        upper_bound <- Q3 + 1.5 * IQR
        return(which(x < lower_bound | x > upper_bound))
      } else {
        return(integer(0))
      }
    },
    
    remove_outliers = function(data, seuil = 0.25) {
      # Only numeric columns
      numeric_cols <- sapply(data, is.numeric)
      data_numeric <- data[, numeric_cols, drop = FALSE]
      
      # Remplacer les outliers par NA
      for (col in colnames(data_numeric)) {
        outlier_indices <- self$detect_outliers(data_numeric[[col]], seuil = seuil)
        data_numeric[outlier_indices, col] <- NA
      }
      
      # Remplacer les colonnes numériques dans les données d'origine
      data[, numeric_cols] <- data_numeric
      
      num_outliers <- sum(is.na(data_numeric))
      cat("Number of outliers replaced with NA:", num_outliers, "\n")
      
      return(data)
    }
    
  )
)

calculate_mode = function(x) {
  uniq_x = unique(x)
  uniq_x[which.max(tabulate(match(x, uniq_x)))]
}

# nolint end