library(shiny)
library(dplyr) # Pour la manipulation de données

# Serveur
server <- function(input, output, session) {
  
  # Réactif pour charger les données
  dataset <- reactive({
    req(input$file) # Assure que le fichier est sélectionné
    
    if (grepl("\\.csv$", input$file$name)) {
      # Lecture du fichier CSV avec détection automatique du délimiteur
      read_delim(input$file$datapath, delim = NULL) 
      
    } else if (grepl("\\.xlsx$", input$file$name)) {
      # Lecture du fichier Excel
      read_excel(input$file$datapath)
    }
  })
  
  # Afficher l'aperçu des données
  output$data_preview <- renderDT({
    req(dataset()) # Assure que les données sont disponibles
    datatable(dataset())
  })
  
  # Générer les UI pour la variable cible
  output$target_var_ui <- renderUI({
    req(dataset()) # Assure que les données sont disponibles
    selectInput("target_var", "Choisir la variable cible", 
                choices = names(dataset()))
  })
  
  # Générer les UI pour les variables explicatives
  output$predictor_vars_ui <- renderUI({
    req(dataset()) # Assure que les données sont disponibles
    checkboxGroupInput("predictor_vars", "Choisir les variables explicatives", 
                       choices = names(dataset()), selected = NULL)
  })
  
  # Affichage des résultats
  output$results <- renderText({
    req(input$calculate) # Assure que le bouton a été cliqué
    "Le résultat s'affiche ici" # Placeholder pour les résultats
  })
}

# Fin du serveur
