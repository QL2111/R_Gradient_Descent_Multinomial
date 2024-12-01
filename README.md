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
8. 
## 🧩 **Introduction**

The `LogisticRegressionMultinomial` class provides a flexible and powerful solution to fit multinomial logistic regression models. The `DataPreparer` class offers advanced data preprocessing, including handling missing values, detecting and removing outliers, and encoding categorical variables.

## Descriptions
This project implements a multinomial logistic regression model using gradient descent, encapsulated in a customizable R package. The package supports mixed predictor variables (qualitative and quantitative) and can be installed directly from GitHub. An interactive Shiny application is included, enabling simplified exploration of the package's features and results.
https://docs.google.com/document/d/156TOP_Mk1kutaAZslE7FNKdw-8WSfL8yQg_orYBqDE0/edit?tab=t.0

## 🛠️ **Installation**
In order to use this package, it is recommend to pass with github
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

On va tester en utilisant le jeu de données : ? Credit card ou student performance ou autres?



🔧 Package Architecture
### The DataPreparer class is responsible for preparing the input data for the logistic regression model.

### The LogisticRegressionMultinomial class is designed to train multinomial logistic regression models using advanced optimization techniques.

💻 Usage Example
```{r}
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


## Multinomial Target Handling

- Cross Entropy

## Authors and License
This project was developed by AWA KARAMOKO, TAHINARISOA DANIELLA RAKOTONDRATSIMBA, QUENTIN LIM as part of the Master 2 SISE program (2024-2025) at Lyon 2 Université Lumière.
Distributed under the MIT License.

🤝 Contributing and Support
Contributions are welcome! Feel free to open an issue or submit a pull request to suggest improvements or report bugs.
- Parallise the computation
- SMOTE


