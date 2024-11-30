library(shiny)
library(DT)


server <- function(input, output, session) {
  
  model <- reactiveVal(NULL)
  results <- reactiveValues()
  
  # Charger les données en fonction du choix de l'utilisateur
  data <- reactive({
    if (input$data_choice == "iris") {
      return(iris)
    } else if (!is.null(input$file)) {
      ext <- tools::file_ext(input$file$name)
      if (ext == "csv") {
        return(read.csv(input$file$datapath))
      } else if (ext == "xlsx") {
        return(read_excel(input$file$datapath))
      }
    }
    return(NULL)
  })
  
  # Mettre à jour la sélection de la variable cible en fonction des données chargées
  observe({
    dataset <- data()
    if (!is.null(dataset)) {
      updateSelectInput(session, "target_var", choices = names(dataset), selected = names(dataset)[ncol(dataset)])
    }
  })
  
  
  # Afficher le tableau des données importées dans la page Import Data
  output$data_table <- DT::renderDT({
    req(data())  # Assurez-vous que les données sont disponibles avant d'afficher le tableau
    datatable(data(), options = list(
      pageLength = 10,  # Limite le nombre de lignes visibles par page
      scrollX = TRUE,   # Permet de défiler horizontalement si le tableau est large
      autoWidth = TRUE, # Ajuste automatiquement la largeur des colonnes
      scrollY = "300px", # Limite la hauteur du tableau avec une barre de défilement vertical
      dom = 'Bfrtip',   # Permet d'ajouter des boutons (tels que des options de tri)
      buttons = c('copy', 'csv', 'excel') # Permet de copier, exporter en CSV ou Excel
    ))
  })
  
  
  
  # Générer l'UI pour la sélection de la variable cible
  output$target_var_ui <- renderUI({
    req(data())
    selectInput("target_var", "Target Variable:", choices = names(data()), selected = names(data())[ncol(data())])
  })
  
  observeEvent(input$train, {
    req(data())
    dataset <- data()
    target_var <- input$target_var
    X <- dataset[, !names(dataset) %in% target_var]
    y_numeric <- as.factor(dataset[[target_var]])
    
    # Vérifier si la variable cible a plus d'une classe
    if (length(unique(y_numeric)) < 2) {
      showNotification("La variable cible doit contenir au moins deux classes.", type = "error")
      return(NULL)
    }
    
    
    # Vérifier les valeurs manquantes
    if (any(is.na(X)) || any(is.na(y_numeric))) {
      showNotification("Les données contiennent des valeurs manquantes. Veuillez les traiter avant de continuer.", type = "error")
      return(NULL)
    }
    
    # Split des données en ensembles d'entraînement et de test
    set.seed(123)  # Pour la reproductibilité
    train_indices <- sample(seq_len(nrow(X)), size = 0.7 * nrow(X))
    X_train <- X[train_indices, ]
    y_train <- y_numeric[train_indices]
    X_test <- X[-train_indices, ]
    y_test <- y_numeric[-train_indices]
    
    # Préparer les données avec DataPreparer
    data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
    prepared_X_train <- data_prep$prepare_data(X_train)
    prepared_X_test <- data_prep$prepare_data(X_test)
    X_train_matrix <- as.matrix(prepared_X_train)
    X_test_matrix <- as.matrix(prepared_X_test)
    
    logistic_model <- LogisticRegressionMultinomial$new(
      learning_rate = input$learning_rate, 
      num_iterations = input$num_iterations,
      optimizer = input$optimizer,
      patience = input$patience,
      use_early_stopping = TRUE
    )
    logistic_model$fit(X_train_matrix, y_train)
    model(logistic_model)  # Stocker le modèle
    
    # Générer les prédictions
    predictions <- logistic_model$predict(X_test_matrix)
    predictions <- factor(predictions, levels = levels(factor(y_test)))  # Align levels with y_test
    
    print(levels(factor(y_test)))
    print(levels(factor(predictions)))
    
    
    confusion_matrix <- table(Predicted = predictions, Actual = factor(y_test))
    results$conf_matrix <- confusion_matrix
    
    # Stocker la courbe de perte
    results$loss_plot <- logistic_model$loss_history
    
    # Stocker les métriques
    results$metrics <- logistic_model$print(X_test_matrix, y_test)
    
  })
  
  # Afficher le résumé
  output$summary_output <- renderPrint({
    req(model())
    model()$summary()
  })
  
  # Afficher les métriques
  output$metrics_output <- renderPrint({
    req(results$metrics)
    cat(results$metrics, sep = "\n")
  })
  
  # Afficher la courbe de perte
  output$loss_plot <- renderPlot({
    req(results$loss_plot)
    plot(results$loss_plot, type = "l", col = "blue", lwd = 2,
         main = "Loss Function Convergence",
         xlab = "Iterations", ylab = "Loss")
  })
  
  # Afficher les courbes ROC
  output$roc_plot <- renderPlot({
    req(model(), data())
    
    dataset <- data()
    target_var <- input$target_var
    X <- dataset[, !names(dataset) %in% target_var]
    y <- as.factor(dataset[[target_var]])
    
    # Préparer les données
    data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
    prepared_X <- data_prep$prepare_data(X)
    X_matrix <- as.matrix(prepared_X)
    
    probabilities <- model()$predict_proba(X_matrix)
    
    # Tracer les courbes ROC
    plot_colors <- rainbow(ncol(probabilities))  # Couleurs pour les classes
    for (i in 1:ncol(probabilities)) {
      binary_response <- as.numeric(y == levels(y)[i])
      roc_curve <- roc(binary_response, probabilities[, i])
      if (i == 1) {
        plot(roc_curve, col = plot_colors[i], main = "ROC Curves", lwd = 2)
      } else {
        plot(roc_curve, col = plot_colors[i], add = TRUE, lwd = 2)
      }
    }
    legend("bottomright", legend = levels(y), col = plot_colors, lwd = 2, title = "Classes")
  })
  
  # Afficher les valeurs AUC
  output$auc_values <- renderPrint({
    req(model(), data())
    
    dataset <- data()
    target_var <- input$target_var
    X <- dataset[, !names(dataset) %in% target_var]
    y <- as.factor(dataset[[target_var]])
    
    # Préparer les données
    data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
    prepared_X <- data_prep$prepare_data(X)
    X_matrix <- as.matrix(prepared_X)
    
    probabilities <- model()$predict_proba(X_matrix)
    
    # Calculer les valeurs AUC
    auc_values <- numeric(ncol(probabilities))
    for (i in 1:ncol(probabilities)) {
      binary_response <- as.numeric(y == levels(y)[i])
      roc_curve <- roc(binary_response, probabilities[, i])
      auc_values[i] <- auc(roc_curve)
    }
    
    # Afficher les valeurs AUC
    cat("Valeurs AUC pour chaque classe :\n")
    for (i in 1:length(auc_values)) {
      cat("Classe", levels(y)[i], ":", auc_values[i], "\n")
    }
  })
}
