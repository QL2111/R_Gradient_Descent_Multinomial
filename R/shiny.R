library(shiny)
library(R6)
library(caret)
library(pROC)
# nolint start
#'@RUN shiny::runApp("R/shiny.R")

# Charger ou définir la classe LogisticRegressionMultinomial ici
source("LogisticRegressionMultinomial.R")
source("DataPreparer.R")

# Interface utilisateur
ui <- fluidPage(
  titlePanel("Logistic Regression Multinomial"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV or XLSX File", accept = c(".csv", ".xlsx")),
      radioButtons("data_choice", "Choose Data Source:",
                   choices = list("Use Iris Dataset" = "iris", "Use Uploaded File" = "uploaded"),
                   selected = "iris"),
      uiOutput("target_var_ui"),  # Dynamic UI for target variable selection
      numericInput("learning_rate", "Learning Rate:", 0.01, min = 0.001, max = 1, step = 0.001),
      numericInput("num_iterations", "Number of Iterations:", 1000, min = 100, max = 10000, step = 100),
      selectInput("optimizer", "Optimizer:", choices = c("adam", "sgd")),
      numericInput("patience", "Early Stopping Patience:", 10, min = 1, max = 100, step = 1),
      actionButton("train", "Train Model")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Summary", verbatimTextOutput("summary_output")),
        tabPanel("Metrics", verbatimTextOutput("metrics_output")),
        tabPanel("Loss Plot", plotOutput("loss_plot")),
        tabPanel("ROC AUC", plotOutput("roc_plot"), verbatimTextOutput("auc_values"))
      )
    )
  )
)

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

    # encode y_test                           ####Changer les levels ? Répréesentation en 1,2,3 mais plsu tard garder les labels?
    y_test <- as.numeric(y_test)
    print(y_test)

    
    # Préparer les données avec DataPreparer
    data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
    prepared_X_train <- data_prep$prepare_data(X_train, target_col = target_var, split_ratio = 0.7, stratify = TRUE)
    prepared_X_test <- data_prep$prepare_data(X_test, target_col = target_var, split_ratio = 0.7, stratify = TRUE)
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
    confusion_matrix <- table(Predicted = predictions, Actual = factor(y_test))
    results$conf_matrix <- confusion_matrix
    
    # Stocker la courbe de perte
    results$loss_plot <- logistic_model$loss_history

    # Stocker les métriques
    results$metrics <- logistic_model$print(X_test_matrix, y_test)

    # Stocker les courbes ROC AUC
    # results$roc_auc <- logistic_model$plot_auc(X_test_matrix, y_test, logistic_model$predict_proba(X_test_matrix))
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
  
  # Afficher les courbes ROC AUC
  output$roc_plot <- renderPlot({
    req(model())
    dataset <- data()
    target_var <- input$target_var
    X <- dataset[, !names(dataset) %in% target_var]
    y <- as.factor(dataset[[target_var]])
    
    # Préparer les données avec DataPreparer
    data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
    prepared_X <- data_prep$prepare_data(X, y, split_ratio = 0.7, stratify = TRUE)
    X_matrix <- as.matrix(prepared_X)
    
    probabilities <- model()$predict_proba(X_matrix)
    
    # Calculer et tracer les courbes ROC
    auc_values <- numeric(ncol(probabilities))
    for (i in 1:ncol(probabilities)) {
      binary_response <- as.numeric(y == levels(y)[i])
      roc_curve <- roc(binary_response, probabilities[, i])
      auc_values[i] <- auc(roc_curve)
      plot(roc_curve, main = paste("ROC Curve for Class", levels(y)[i]), col = i, add = i != 1)
    }
  })
  
  # Afficher les valeurs AUC
  output$auc_values <- renderPrint({
    req(model())
    dataset <- data()
    target_var <- input$target_var
    X <- dataset[, !names(dataset) %in% target_var]
    y <- as.factor(dataset[[target_var]])
    
    # Préparer les données avec DataPreparer
    data_prep <- DataPreparer$new(use_factor_analysis = FALSE)
    prepared_X <- data_prep$prepare_data(X, target_col = target_var, split_ratio = 0.7, stratify = TRUE)
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
    cat("AUC values for each class:\n")
    for (i in 1:length(auc_values)) {
      cat("Class", levels(y)[i], ":", auc_values[i], "\n")
    }
  })
}

shinyApp(ui = ui, server = server)

# nolint end
