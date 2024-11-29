# DataPreparer.R
# Générer doc
# devtools::document("D:/GitHub/R_Gradient_Descent_Multinomial")
# nolint start
#' @title Data Preparation Class
#' @description The `DataPreparer` class provides methods to standardize quantitative data and encode qualitative data, allowing the option of factor analysis for mixed data.
#' @details This class is part of a package designed to support data preparation tasks, particularly for models that need standardized quantitative features and encoded categorical features. The class can handle mixed data types and offers both one-hot encoding and an alternative factor analysis encoding for qualitative data.
#' 
#' @field use_factor_analysis A logical flag indicating whether to apply factor analysis on qualitative data instead of one-hot encoding. Default is \code{FALSE}.
#' 
#' @export
DataPreparer <- R6::R6Class("DataPreparer", 
  public = list(
    #' @field use_factor_analysis Logical. Indicates whether to apply factor analysis for qualitative data encoding.
    use_factor_analysis = FALSE,
    
    #' @description Initializes a new instance of the `DataPreparer` class.
    #' @param use_factor_analysis Logical. Set to `TRUE` to enable factor analysis for qualitative variables, `FALSE` for one-hot encoding.
    #' @return A new `DataPreparer` object.
    initialize = function(use_factor_analysis = FALSE) {
      self$use_factor_analysis <- use_factor_analysis
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
    prepare_data = function(data) {
      quantitative_vars <- sapply(data, is.numeric) # Check the type isnumeric
      qualitative_vars <- !quantitative_vars # Check the type is not numeric
      prepared_list <- list()
      
      # Process quantitative variables: standardization
      if (any(quantitative_vars)) {
        quant_data <- data[, quantitative_vars, drop = FALSE] # drop=FALSE to keep data frame or else it will transform into a vector
        quant_data <- as.data.frame(lapply(quant_data, self$standardize)) # Apply the standardize function
        prepared_list$quantitative <- quant_data
      }
      
      # Process qualitative variables: factor analysis or one-hot encoding
      if (any(qualitative_vars)) {
        qual_data <- data[, qualitative_vars, drop = FALSE]
        
        if (self$use_factor_analysis) { # IF use_factor_analysis TRUE use factor_analysis instead of one-hot encoding
          # Apply factor analysis for qualitative data
          qual_data <- factor_analysis_mixed(qual_data)
        } else {
          # One-hot encoding
          qual_data <- as.data.frame(model.matrix(~ . - 1, data = qual_data)) # ~ . - 1 means all columns except the first one(exclude intercept - to avoid multicollinearity)
        }
        
        prepared_list$qualitative <- qual_data
      }
      
      # Combine quantitative and qualitative data
      prepared_data <- do.call(cbind, prepared_list)
      return(prepared_data)
    }
  )
)

# nolint
