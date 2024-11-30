# R_Gradient_Descent_Multinomial

## Descriptions
This project implements a multinomial logistic regression model using gradient descent, encapsulated in a customizable R package. The package supports mixed predictor variables (qualitative and quantitative) and can be installed directly from GitHub. An interactive Shiny application is included, enabling simplified exploration of the package's features and results.
https://docs.google.com/document/d/156TOP_Mk1kutaAZslE7FNKdw-8WSfL8yQg_orYBqDE0/edit?tab=t.0

## Installation and Data loading
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



## Architecture


## Usage


## Multinomial Target Handling

## Authors and License
This project was developed by AWA KARAMOKO, TAHINARISOA DANIELLA RAKOTONDRATSIMBA, QUENTIN LIM as part of the Master 2 SISE program (2024-2025) at Lyon 2 Université Lumière.
Distributed under the MIT License.


