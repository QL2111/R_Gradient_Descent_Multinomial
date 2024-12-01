library(shiny)
library(DT)
library(Metrics)  # Pour calculer l'AUC

# source("R/LogisticRegressionMultinomial.R")
# source("R/DataPrerarer.R")

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
  
  # Mettre à jour la sélection de la variable cible
  observe({
    dataset <- data()
    if (!is.null(dataset)) {
      updateSelectInput(session, "target_var", choices = names(dataset), selected = names(dataset)[ncol(dataset)])
    }
  })
  
  # Afficher le tableau des données importées
  output$data_table <- DT::renderDT({
    req(data())
    datatable(data(), options = list(
      pageLength = 10,
      scrollX = TRUE,
      autoWidth = TRUE,
      scrollY = "300px",
      dom = 'Bfrtip',
      buttons = c('copy', 'csv', 'excel')
    ))
  })
  
  # Générer l'UI pour la sélection de la variable cible
  output$target_var_ui <- renderUI({
    req(data())
    selectInput("target_var", "Variable cible:", choices = names(data()), selected = names(data())[ncol(data())])
  })
  
  # Fonction pour préparer les données
  prepare_data <- function(X, y) {
    if (!is.matrix(X)) {
      X <- as.matrix(X)
    }
    if (nrow(X) != length(y)) {
      stop("Le nombre de lignes de X ne correspond pas à la longueur de y")
    }
    X <- cbind(1, X)  # Ajout de la colonne de biais
    return(X)
  }
  
  observeEvent(input$train, {
    req(data())
    dataset <- data()
    target_var <- input$target_var
    X <- dataset[, !names(dataset) %in% target_var]
    y <- as.factor(dataset[[target_var]])
    
    # Vérification des valeurs manquantes
    if (any(is.na(X)) || any(is.na(y))) {
      showNotification("Les données contiennent des valeurs manquantes. Veuillez les traiter avant de continuer.", type = "error")
      return(NULL)
    }
    
    # Division des données en ensembles d'entraînement et de test
    set.seed(123)
    train_indices <- sample(seq_len(nrow(X)), size = 0.7 * nrow(X))
    X_train <- X[train_indices, ]
    y_train <- y[train_indices]
    X_test <- X[-train_indices, ]
    y_test <- y[-train_indices]
    
    # Préparer les données
    X_train_prepared <- prepare_data(X_train, y_train)
    X_test_prepared <- prepare_data(X_test, y_test)
    
    # Entraînement du modèle de régression logistique multinomiale
    logistic_model <- LogisticRegressionMultinomial$new(
      learning_rate = input$learning_rate,
      num_iterations = input$num_iterations,
      optimizer = input$optimizer,
      regularization = input$regularization,
      patience = input$patience,
      use_early_stopping = TRUE
    )
    logistic_model$fit(X_train_prepared, y_train)
    model(logistic_model)  # Stocker le modèle pour les étapes suivantes
    
    # Générer les prédictions
    predictions <- logistic_model$predict(X_test_prepared)
    
    # Confusion matrix et métriques
    confusion_matrix <- table(Predicted = predictions, Actual = y_test)
    results$conf_matrix <- confusion_matrix
    results$metrics <- logistic_model$print(X_test_prepared, y_test)
    results$loss_plot <- logistic_model$loss_history
  })
  
  # Afficher le résumé du modèle
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
    X_prepared <- prepare_data(X, y)
    probabilities <- model()$predict_proba(X_prepared)
    
    # Tracer les courbes ROC
    plot_colors <- rainbow(ncol(probabilities))
    for (i in 1:ncol(probabilities)) {
      binary_response <- as.numeric(y == levels(y)[i])
      roc_curve <- roc(binary_response, probabilities[, i])
      if (i == 1) {
        plot(roc_curve, col = plot_colors[i], main = "ROC Curves", lwd = 2)
      } else {
        plot(roc_curve, col = plot_colors[i], add = TRUE, lwd = 2)
      }
    }
    legend("bottomright", legend = levels(y), fill = plot_colors)
  })
  
  # Afficher l'AUC
  output$auc_values <- renderPrint({
    req(model(), data())
    dataset <- data()
    target_var <- input$target_var
    X <- dataset[, !names(dataset) %in% target_var]
    y <- as.factor(dataset[[target_var]])
    
    # Préparer les données
    X_prepared <- prepare_data(X, y)
    probabilities <- model()$predict_proba(X_prepared)
    
    auc_values <- sapply(1:ncol(probabilities), function(i) {
      binary_response <- as.numeric(y == levels(y)[i])
      auc(binary_response, probabilities[, i])
    })
    cat("AUC Values:\n")
    print(auc_values)
  })
}

