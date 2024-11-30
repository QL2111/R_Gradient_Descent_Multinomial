library(shiny)

# Interface pour l'onglet Analyse
analyse_ui <- fluidPage(
  h2("Paramètres du Modèle"),
  
  # Utilisation de tabsetPanel pour organiser en plusieurs panneaux
  tabsetPanel(
    
    # Premier panneau : Paramètres, sélection des variables, et informations
    tabPanel(
      title = "Configuration",
      
      # Disposition des éléments en deux colonnes
      fluidRow(
        
        # Colonne de gauche : Sélection des variables et ajustement des paramètres
        column(
          width = 6,
          wellPanel(
            h4("Sélection des variables"),
            uiOutput("variable_select_ui"), # Interface dynamique pour sélectionner les variables
            actionButton("apply_calcul", "Lancer le calcul")
          ),
          wellPanel(
            h4("Ajuster les paramètres du modèle"),
            numericInput("param1", "Learning rate", value = 0.01, step = 0.01),
            numericInput("param2", "Num itérations", value = 100, step = 1),
            actionButton("apply_params", "Appliquer")
          )
        ),
        
        # Colonne de droite : Informations et résumé du modèle
        column(
          width = 6,
          wellPanel(
            h4("Informations sur le modèle"),
            textOutput("model_info")  # Zone pour afficher les informations du modèle
          ),
          wellPanel(
            h4("Résumé du modèle"),
            verbatimTextOutput("model_summary")  # Zone pour afficher le résumé du modèle
          )
        )
      )
    ),
    
    # Deuxième panneau : Graphiques et diagnostics
    tabPanel(
      title = "Graphiques",
      
      # Zone pour afficher les graphiques du modèle
      wellPanel(
        h4("Graphiques"),
        plotOutput("model_diagnostics", height = "400px")  # Placeholder pour un graphique
      )
    )
  )
)
