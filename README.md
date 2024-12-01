# R_Gradient_Descent_Multinomial

## 📖 **Table of Contents**

1. [Introduction](#introduction)
2. [Descriptions](#descriptions)
3. [Installation](#installation)
4. [Package Architecture](#package-architecture)
   * [DataPreparer: Data Preparation](#datapreparer-data-preparation)
   * [LogisticRegressionMultinomial: Logistic Regression Model](#logisticregressionmultinomial-logistic-regression-model)
5. [Data Flow](#data-flow)
6. [Usage Example](#usage-example)
7. [Output](#output)
8. [Multinomial Target Handling](#multinomial-target-handling)
9. [Authors and License](#authors-and-license)
10. [Contributing and Support](#contributing-and-support)

<h2 id="introduction">🧩 Introduction</h2>

The `LogisticRegressionMultinomial` class provides a flexible and powerful solution to fit multinomial logistic regression models. The `DataPreparer` class offers advanced data preprocessing and should be used in accordance with the `LogistRegressionMultinomial` class. This project is in accordance with the courses "Programmation en R" dispensed in Lyon 2 Université Lumière Master 2 -SISE.

<h2 id="descriptions">🤖 Descriptions</h2>

This project implements a multinomial logistic regression model using gradient descent, encapsulated in a customizable R package. The package supports mixed predictor variables (qualitative and quantitative) and can be installed directly from GitHub. An interactive Shiny application is included, enabling simplified exploration of the package's features and results. By default if no dataset is loaded, it will load the `Iris`dataset. The DataPreparer can then support the data preparation, once the data is ready, the LogisticRegressionMultinomial model can fit the data.

<h2 id="installation">🛠️ Installation</h2>

In order to use this package, it is recommend to use `devtools::install_github`.
```r
library(devtools)
devtools::install_github("QL2111/R_Gradient_Descent_Multinomial")

```

We will now load the library M2LogReg.
```r
library(M2LogReg)
```

Let's check if it's correctly loaded.

```r
help("DataPreparer")
```
![Extrait Documentation DataPreparer](/images/Extrait_doc_DataPreparer.png)

We also support installation with an exported file available on [Google Drive](https://drive.google.com/drive/folders/1uZ6iTvHueYE0HFiWNZPQYNUN59fZ8HGi?usp=sharing).

```r
library(devtools)
install.packages([Specify your path here], repos = NULL, type = "source")

```

This project also include a R Shiny app, we can run it by using :
```
library(shiny)
# From the R_Gradient_Descent_Multinomial folder
runApp("Shiny/app.R")

```

<h2 id="package-architecture">🔧 Package Architecture</h2>

### **DataPreparer: Data Preparation**

The `DataPreparer` class is responsible for preparing the input data for the logistic regression model.

#### **Public Fields**

| Field                | Description                                                                                   |
|----------------------|-----------------------------------------------------------------------------------------------|
| `use_factor_analysis` | Logical. Indicates whether to apply factor analysis for both quantitative and qualitative variables.|
| `cumulative_var_threshold` | The threshold for cumulative variance explained by factor analysis. Default is 90.|


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

<h2 id="data-flow">🏗️ Data Flow</h2>

1. **Data Loading and Preparation**  

   * Detect and removes outliers with the chosen threshold, using the IQR method.  
   * Handle missing values through imputation (median for quantitative variables and mode for categorical variables).  
   * Train/Test split with the option to stratify.
   * Use ont-hot encoding of the categoricals values.
   * Standardization of the numericals values.
   * Use a factor analysis if specified and only keep the number of the dimensions that reach a threshold of cumulative variance.

2. **Model Training**  
   * Initialize model coefficients and set hyperparameters such as learning rate, number of iterations, loss function, early stopping, patience values, batch size and regularization method.  
   * The loss function ny default is the Cross Entropy Loss.
   * We will use softmax to project the values.
   * Train the model using gradient descent optimization methods such as Adam or Stochastic Gradient Descent (SGD).
   * The early stopping approach will use a validation set and stop if there is no improvement in the loss for a certain number of iterations in a row(patience counter).
   * The Adam Optimizer use a mini-batch approach.
   * We will use a regularization approach to avoid overfitting (L1, L2, or ElasticNet).
   * Track the loss function at each iteration and store it in `loss_history`.

3. **Prediction and Evaluation**  
   * Predict the class labels or probabilities for new data using `predict()` or `predict_proba()`.  
   * Evaluate model performance using confusion matrices, F1-score and accuracy.
   * We can also plot the evoluation of the loss function, the ROC and AUC values for each class.
   * Display feature importance with a barplot.
   * The user can also export the model in a pmml format with the coefficients.

<h2 id="usage-example">💻 Usage Example</h2>

We will test this package with the `IRIS` dataset

```r

# Load example data
data(iris)
X <- iris[, -5]  # Predictor variables
y <- iris$Species  # Target variable

# Prepare the data
data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
prepared_data <- data_prep$prepare_data(iris, target_col = "Species")
X_train <- prepared_data$X_train
X_test <- prepared_data$X_test
y_train <- prepared_data$y_train
y_test <- prepared_data$y_test

X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)


# Convert the target variable to numeric values
y_train_numeric <- as.numeric(y_train)
y_test_numeric <- as.numeric(y_test)

# Initialize the model
model <- LogisticRegressionMultinomial$new(
  learning_rate = 0.01,
  num_iterations = 500,
  optimizer = "adam",
  batch_size = 32,
  use_early_stopping = TRUE,
  regularization = "ridge"
)

model$fit(X_train_matrix, y_train_numeric)

# Prédire sur l'ensemble de test
predictions <- model$predict(X_test_matrix)


# Display the model summary(parameters)
model$summary()

# Display performance metrics
model$print(X_test_matrix, y_test_numeric)

# Display variable importance
model$var_importance()

# Display the loss plot
model$plot_loss()

# Display n(5) best variables
model$select_variables(5)

```
<h2 id="Rshiny App">✔️ Rshiny App</h2>
Pour lancer l'application, ..............

<h2 id="output">📊 Example Output</h2>

Mettre les graphiques de loss, auc, var importances
Montrer le summary et le print depuis le RShiny(capture d'écran)

<h2 id="multinomial-target-handling">🎯 Multinomial Target Handling</h2>

 One of the challenge of this class was to support Multinomial prediction.
 Unlike binary logistic regression, which predicts probabilities for only two classes, this class computes probabilities across multiple classes by converting the logits into normalized probabilities. It uses one-hot encoding to represent the target variable and minimizes a `multinomial cross-entropy loss (log-loss)` during training. The class supports advanced optimization methods like Adam and Stochastic Gradient Descent (SGD), allowing it to efficiently update the coefficients for all classes simultaneously. This enables the model to learn complex decision boundaries across multiple classes, making it well-suited for multinomial classification tasks.

<h2 id="authors-and-license">⚖️ Authors and License</h2>

This project was developed by AWA KARAMOKO, TAHINARISOA DANIELLA RAKOTONDRATSIMBA et QUENTIN LIM as part of the Master 2 SISE program (2024-2025) at Lyon 2 Université Lumière.
Distributed under the MIT License.

<h2 id="contributing-and-support">🤝 Contributing and Support</h2>

Contributions are welcome! Feel free to open an issue or submit a pull request to suggest improvements or report bugs.
You can certainly contribute by working on these topics :
- Implementing SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced datasets.
- Parallelizing the computations for the mini-batch approach to improve training efficiency and speed.
- Enhancing the visualization capabilities for model evaluation.
- Improve the UI.
- Continue the interconnection between the package and the Rshiny using Docker.
- Add error messages to report unspecified parameters on the shiny App.

