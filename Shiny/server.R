library(shiny)
library(readr)
library(readxl)

# Charger le script contenant la classe LogisticRegressionMultinomial
source("LogisticRegressionMultinomial.R")

# Serveur principal de l'application
server <- function(input, output, session) {
  
  # Réactif pour stocker les données importées
  data <- reactive({
    req(input$file)
    ext <- tools::file_ext(input$file$name)
    if (ext == "csv") {
      read_delim(input$file$datapath, delim = input$delimiter)
    } else if (ext %in% c("xls", "xlsx")) {
      read_excel(input$file$datapath)
    } else {
      NULL
    }
  })
  
  # Affichage du tableau de données
  output$data_table <- renderTable({
    req(data())
    data()
  }, rownames = TRUE)
  
  # Mettre à jour les choix de variables en fonction des données importées
  observe({
    req(data())
    updateSelectInput(session, "var_to_plot", choices = names(data()), selected = names(data())[1])
  })
  
  # Graphique dynamique pour explorer les données
  output$dynamic_plot <- renderPlot({
    req(data(), input$var_to_plot)
    var <- input$var_to_plot
    var_data <- data()[[var]]
    if (is.numeric(var_data)) {
      hist(var_data, main = paste("Distribution de", var), xlab = var, col = "#26B67B", border = "white")
    } else if (is.factor(var_data) || is.character(var_data)) {
      barplot(table(var_data), main = paste("Distribution de", var), xlab = var, col = "orange", border = "white")
    } else {
      plot.new()
      text(0.5, 0.5, "Type de variable non supporté", cex = 1.5)
    }
  })
  
  # UI pour sélectionner les variables
  output$variable_select_ui <- renderUI({
    req(data())
    column_names <- names(data())
    tagList(
      selectInput("target_var", "Sélectionnez la variable cible", choices = column_names),
      selectInput("predictor_vars", "Sélectionnez les variables explicatives", choices = column_names, multiple = TRUE)
    )
  })
  
  # Réactifs pour le modèle et son résumé
  reactive_model <- reactiveVal(NULL)
  reactive_summary <- reactiveVal(NULL)
  
  # Lancer le calcul lorsque le bouton est cliqué
  observeEvent(input$apply_calcul, {
    req(data(), input$target_var, input$predictor_vars)
    
    # Instancier un objet de la classe LogisticRegressionMultinomial
    model <- LogisticRegressionMultinomial$new(
      data = data(),
      target_var = input$target_var,
      predictor_vars = input$predictor_vars
    )
    
    # Enregistrer le modèle et son résumé
    reactive_model(model)
    reactive_summary(model$summary())  # Appel de la méthode summary de la classe
  })
  
  # Afficher les informations sur le modèle
  output$model_info <- renderText({
    model <- reactive_model()
    if (is.null(model)) return("Aucun modèle généré.")
    paste("Modèle ajusté avec", length(model$coefficients), "coefficients.")
  })
  
  # Afficher le résumé du modèle
  output$model_summary <- renderPrint({
    reactive_summary()
  })
  
  # Générer les prédictions lorsque le bouton "Lancer le calcul" est cliqué
  output$predictions <- renderTable({
    req(reactive_model())  # Vérifie que le modèle a été créé
    model <- reactive_model()
    
    # Effectuer les prédictions avec les données d'entrée
    predictions <- model$predict(newdata = data())
    
    # Afficher les prédictions dans un tableau
    data.frame(Predictions = predictions)
  })
}
