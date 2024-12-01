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
      menuItem("Résultats et métriques", tabName = "results", icon = icon("chart-bar")),
      menuItem("Visualisations", tabName = "plots", icon = icon("pie-chart"))
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
            title = "Importer les données", width = 12,
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
            title = "Paramètres du modèle", width = 12,
            solidHeader = TRUE,
            uiOutput("target_var_ui"),
            numericInput("learning_rate", "Taux d'apprentissage:", 0.01, min = 0.001, max = 1, step = 0.001),
            numericInput("num_iterations", "Nombre d'itérations:", 1000, min = 100, max = 10000, step = 100),
            selectInput("optimizer", "Optimiseur:", choices = c("adam", "sgd")),
            selectInput("regularization", "Regularisation:", choices = c("none", "lasso", "ridge", "elasticnet")),
            numericInput("patience", "Patience pour early stopping:", 10, min = 1, max = 100, step = 1),
            numericInput("batch_size", "Batch size:", 32),
          )
        ),
        
        fluidRow(
          box(
            title = "Paramètres - Preprocessing", width = 12,
            solidHeader = TRUE,
            checkboxInput("use_factor_analysis", "Utiliser l'analyse factorielle", value = FALSE),
            numericInput("split_ratio", "Taux de ratio (Entraînement/Validation)", value = 0.7, min = 0.5, max = 0.9, step = 0.05),
            checkboxInput("remove_outliers", "Supprimer les outliers", value = FALSE),
            numericInput("outlier_threshold", "Seuil pour les outliers", value = 0.25, min = 0.01, max = 0.5, step = 0.01),
            checkboxInput("stratify", "Stratify", value = FALSE)
          )
        ),
        
        # Centrer le bouton "Entraîner le Modèle"
        fluidRow(
          column(
            width = 12,
            class = "text-center",  # Centrer le contenu de la colonne
            actionButton("train", "Entraîner le Modèle", width = "200px")  # Centrer le bouton
          )
        )
      ),
      
      # Page Résultats
      tabItem(
        tabName = "results",
        fluidRow(
          box(
            title = "Hyperparamètres", width = 6,
            solidHeader = TRUE, verbatimTextOutput("summary_output")
          ),
          box(
            title = "Métriques", width = 6,
            solidHeader = TRUE, verbatimTextOutput("metrics_output")
          )
        ),
        fluidRow(
          column(
            width = 12,
            class = "text-center",
            actionButton("exportpmml", "Exporter le modèle en PMML"),
            tags$div(style = "margin-top: 20px;", uiOutput("download_link"))
          )
        )
      ),
      
      # Page Visualisation
      tabItem(
        tabName = "plots",
        fluidRow(
          # Premier bloc pour la courbe de perte
          box(
            title = "Loss", width = 12,  # Utilisation de width = 12 pour occuper toute la largeur
            solidHeader = TRUE, plotOutput("loss_plot")
          )
        ),
        fluidRow(
          # Deuxième bloc pour le ROC AUC
          box(
            title = "ROC AUC", width = 12,  # Utilisation de width = 12 pour occuper toute la largeur
            solidHeader = TRUE, plotOutput("roc_plot"), verbatimTextOutput("auc_values")
          )
        ),
        fluidRow(
          # Troisième bloc pour importance des variables
          box(
            title = "Importance des variables", width = 12, height = "auto",  # Utilisation de width = 12 pour occuper toute la largeur
            solidHeader = TRUE, plotOutput("var_imp_plot")  # Remplacez "other_plot" par votre propre graphique
          )
        )
      )
    )
  )
)


