# R_Gradient_Descent_Multinomial

## 📖 **Table of Contents**

1. [Introduction](#introduction)  
2. [Installation](#installation)  
3. [Package Architecture](#package-architecture)  
   * [DataPreparer: Data Preparation](#datapreparer-data-preparation)  
   * [LogisticRegressionMultinomial: Logistic Regression Model](#logisticregressionmultinomial-logistic-regression-model)  
4. [Usage Example](#usage-example)
5. [Multinomial Target Handling](#Multinomial Target Handling)
6. [Features and Methods](#features-and-methods)  
7. [Contributing and Support](#contributing-and-support)

## 🧩 **Introduction**

The `LogisticRegressionMultinomial` class provides a flexible and powerful solution to fit multinomial logistic regression models. The `DataPreparer` class offers advanced data preprocessing, including handling missing values, detecting and removing outliers, and encoding categorical variables.

## 🤖 Descriptions
This project implements a multinomial logistic regression model using gradient descent, encapsulated in a customizable R package. The package supports mixed predictor variables (qualitative and quantitative) and can be installed directly from GitHub. An interactive Shiny application is included, enabling simplified exploration of the package's features and results.
https://docs.google.com/document/d/156TOP_Mk1kutaAZslE7FNKdw-8WSfL8yQg_orYBqDE0/edit?tab=t.0

## 🛠️ **Installation**
In order to use this package, it is recommend to use `devtools::install_github`
```r
library(devtools)
devtools::install_github("QL2111/R_Gradient_Descent_Multinomial")

```

We will now load the library M2LogReg
```r
library(M2LogReg)
```

Let's check if it's correctly loaded

```r
help("DataPreparer")
```
![Extrait Documentation DataPreparer](/images/Extrait_doc_DataPreparer.png)

We can also support the installation with an exported file available on :"Mettre lien du google drive".

```r
library(devtools)
install.packages([Specify your path here], repos = NULL, type = "source")

```


## 🔧 Package Architecture
### **DataPreparer: Data Preparation**

The `DataPreparer` class is responsible for preparing the input data for the logistic regression model.

#### **Public Fields**

| Field                | Description                                                                                   |
|----------------------|-----------------------------------------------------------------------------------------------|
| `use_factor_analysis` | Logical. Indicates whether to apply factor analysis for both quantitative and qualitative variables.|

#### **Public Methods**

| Method                   | Description                                                                                     |
|--------------------------|-------------------------------------------------------------------------------------------------|
| **`new()`**               | Initializes a new instance of the `DataPreparer` class.                                         |
| **`calculate_mode()`**    | Calculates the mode of a categorical variable.                                                  |
| **`standardize()`**       | Standardizes a quantitative variable (mean = 0, standard deviation = 1).                        |
| **`split_data()`**        | Splits the data into training and testing sets.                                                 |
| **`prepare_data()`**      | Prepares the data by standardizing quantitative variables and encoding categorical variables.    |
| **`handle_missing_data()`**| Handles missing data by imputing median values for quantitative variables and mode for categorical ones.|
| **`process_data()`**      | Processes the data globally (standardization, encoding, and optional factor analysis).           |
| **`detect_outliers()`**   | Detects outliers in the data using the IQR method.                                              |
| **`remove_outliers()`**   | Removes detected outliers from the dataset.                                                     |

---

### **LogisticRegressionMultinomial: Logistic Regression Model**

The `LogisticRegressionMultinomial` class is designed to train multinomial logistic regression models using advanced optimization techniques.

#### **Public Fields**

| Field                | Description                                                                                   |
|----------------------|-----------------------------------------------------------------------------------------------|
| `coefficients`       | Matrix of model coefficients, initialized during the `fit()` method.                           |
| `learning_rate`      | Learning rate for gradient descent. Default: `0.01`.                                           |
| `num_iterations`     | Number of iterations for model training. Default: `1000`.                                      |
| `loss_history`       | Numeric vector tracking the loss function at each iteration.                                   |
| `beta1`              | Momentum parameter for the Adam optimizer. Default: `0.9`.                                     |
| `beta2`              | Second momentum parameter for the Adam optimizer. Default: `0.999`.                            |
| `epsilon`            | Small constant for numerical stability. Default: `1e-8`.                                       |
| `use_early_stopping` | Logical. Enables early stopping based on validation loss. Default: `TRUE`.                      |
| `patience`           | Number of iterations to wait for improvement before stopping early. Default: `20`.             |
| `regularization`     | Regularization method. Options: `"none"`, `"ridge"`, `"lasso"`, `"elasticnet"`. Default: `"none"`.|
| `loss_function`      | Loss function used during optimization. Options: `"quadratique"`, `"logistique"`. Default: `"logistique"`.|
| `batch_size`         | Size of the mini-batch for gradient descent. Default: `32`.                                     |

#### **Public Methods**

| Method                        | Description                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------|
| **`new()`**                   | Initializes a new instance of the `LogisticRegressionMultinomial` class.                        |
| **`fit()`**                   | Fits the model to the training data.                                                            |
| **`adam_optimizer()`**        | Implements the Adam optimizer for gradient-based updates.                                       |
| **`sgd_optimizer()`**         | Implements Stochastic Gradient Descent (SGD) optimizer.                                         |
| **`softmax()`**               | Computes the softmax function for multi-class probabilities.                                    |
| **`one_hot_encode()`**        | Converts the response variable into a one-hot encoded matrix.                                   |
| **`predict()`**               | Predicts class labels for new data.                                                             |
| **`predict_proba()`**         | Predicts class probabilities for new data.                                                      |
| **`plot_loss()`**             | Plots the convergence of the loss function during training.                                     |
| **`plot_auc()`**              | Plots the ROC curve and computes the AUC for each class.                                        |
| **`var_importance()`**        | Calculates and displays the importance of each feature based on model coefficients.             |
| **`apply_regularization()`**  | Applies regularization to the model during training.                                            |
| **`export_pmml()`**           | Exports the trained model to PMML format for external use.                                      |

---

## 🏗️ Data Flow

1. **Data Loading and Preparation**  
   * Handle missing values through imputation (median for quantitative variables and mode for categorical variables).  
   * Detect and remove outliers using the IQR method.  
   * Encode categorical variables using either one-hot encoding or factor analysis, depending on the user's preference.

2. **Model Training**  
   * Initialize model coefficients and set hyperparameters such as learning rate, number of iterations, and regularization method.  
   * Train the model using gradient descent optimization methods such as Adam or Stochastic Gradient Descent (SGD).  
   * Track the loss function at each iteration and store it in `loss_history`.

3. **Prediction and Evaluation**  
   * Predict the class labels or probabilities for new data using `predict()` or `predict_proba()`.  
   * Evaluate model performance using confusion matrices, ROC curves, and AUC values for each class.  
   * Display feature importance and the overall model summary.

## 💻 Usage Example
We will test this pacjage with the `IRIS` dataset
```r
# Load the package
library(LogisticRegressionMultinomial)
library(DataPreparer)

# Load example data
data(iris)
X <- iris[, -5]  # Predictor variables
y <- iris$Species  # Target variable

# Prepare the data
data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
prepared_data <- data_prep$prepare_data(iris, target_col = "Species")
X_matrix <- as.matrix(prepared_data[, -1])

# Initialize the model
model <- LogisticRegressionMultinomial$new(
  learning_rate = 0.01,
  num_iterations = 500,
  optimizer = "adam",
  batch_size = 32,
  use_early_stopping = TRUE,
  regularization = "ridge"
)

# Fit the model
model$fit(X_matrix, y)

# Predict new data
predictions <- model$predict(X_matrix)

# Display performance metrics
model$print(X_matrix, y)

# Display variable importance
model$var_importance()
```


## 🎯 Multinomial Target Handling

- Cross Entropy

## ⚖️ Authors and License
This project was developed by AWA KARAMOKO, TAHINARISOA DANIELLA RAKOTONDRATSIMBA, QUENTIN LIM as part of the Master 2 SISE program (2024-2025) at Lyon 2 Université Lumière.
Distributed under the MIT License.

## 🤝 Contributing and Support
Contributions are welcome! Feel free to open an issue or submit a pull request to suggest improvements or report bugs.
- Parallise the computation
- SMOTE


