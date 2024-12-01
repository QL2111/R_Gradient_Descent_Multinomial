library(shiny)
library(shinydashboard)
library(shinymaterial)
library(DT)

ui <- dashboardPage(
  skin = "blue", # Thème bleu pour shinydashboard
  dashboardHeader(title = "Logistic Regression Multinomial"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Importation", tabName = "import", icon = icon("file-upload")),
      menuItem("Paramètres du Modèle", tabName = "parameters", icon = icon("cogs")),
      menuItem("Résultats", tabName = "results", icon = icon("chart-bar")),
      menuItem("Visualisation", tabName = "plots", icon = icon("chart-bar"))
    )
  ),
  dashboardBody(
    # Inclure le fichier CSS
    tags$head(
      tags$link(rel = "stylesheet", type = "text/css", href = "style.css"),
      tags$link(rel = "stylesheet", href = "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css")
    ),
    tabItems(
      # Page Importation
      tabItem(
        tabName = "import",
        fluidRow(
          box(
            title = "Importer les Données", width = 12,
            solidHeader = TRUE,
            fileInput("file", "Télécharger un fichier CSV ou XLSX", accept = c(".csv", ".xlsx")),
            radioButtons("data_choice", "Choisissez la source des données:",
                         choices = list("Utiliser Iris Dataset" = "iris", "Fichier téléchargé" = "uploaded"),
                         selected = "iris")
          )
        ),
        
        # Tableau pour afficher les données importées
        fluidRow(
          box(
            title = "Données Importées", width = 12,
            solidHeader = TRUE,
            DT::dataTableOutput("data_table")  # Afficher le tableau des données importées
          )
        )
        
        
        
        
      ),
      # Page Paramètres du Modèle
      tabItem(
        tabName = "parameters",
        fluidRow(
          box(
            title = "Configurer les Paramètres du Modèle", width = 12,
            solidHeader = TRUE,
            uiOutput("target_var_ui"),
            numericInput("learning_rate", "Taux d'apprentissage:", 0.01, min = 0.001, max = 1, step = 0.001),
            numericInput("num_iterations", "Nombre d'itérations:", 1000, min = 100, max = 10000, step = 100),
            selectInput("optimizer", "Optimiseur:", choices = c("adam", "sgd")),
            selectInput("regularization", "Regularisation:", choices = c("none", "lasso", "ridge", "elasticnet")),
            numericInput("patience", "Patience pour early stopping:", 10, min = 1, max = 100, step = 1),
            actionButton("train", "Entraîner le Modèle")
          )
        )
      ),
      # Page Résultats
      tabItem(
        tabName = "results",
        fluidRow(
          box(
            title = "Résumé", width = 6,
            solidHeader = TRUE, verbatimTextOutput("summary_output")
          ),
          box(
            title = "Métriques", width = 6,
            solidHeader = TRUE, verbatimTextOutput("metrics_output")
          ),
          
        )
      ),
      tabItem(
        tabName = "plots",
        fluidRow(
          box(
            title = "Courbe de Perte", width = 6,
            solidHeader = TRUE, plotOutput("loss_plot")
          ),
          box(
            title = "ROC AUC", width = 6,
            solidHeader = TRUE, plotOutput("roc_plot"), verbatimTextOutput("auc_values")
          )
        )
      )
      
    )
  )
)
