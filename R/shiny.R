library(shiny)
library(R6)
library(caret)
library(pROC)

#'@RUN shiny::runApp("R/shiny.R")

# Charger ou définir la classe LogisticRegressionMultinomial ici
# (Coller le code complet de la classe LogisticRegressionMultinomial ici)
source("LogisticRegressionMultinomial.R")
# Interface utilisateur
ui <- fluidPage(
  titlePanel("Multinomial Logistic Regression Trainer"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV File", accept = ".csv"),
      checkboxInput("use_default", "Use Default Dataset (iris)", value = TRUE),
      numericInput("learning_rate", "Learning Rate", value = 0.01, step = 0.001),
      numericInput("num_iterations", "Number of Iterations", value = 1000, min = 100),
      actionButton("train", "Train Model")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Summary", 
                 verbatimTextOutput("summary")),
        tabPanel("Loss Plot", 
                 plotOutput("loss_plot")),
        tabPanel("Confusion Matrix", 
                 tableOutput("conf_matrix"))
      )
    )
  )
)

# Serveur
server <- function(input, output, session) {
  data <- reactive({
    if (input$use_default) {
      # Utiliser le jeu de données iris comme exemple
      iris$Species <- as.factor(as.numeric(iris$Species) - 1) # Convertir en binaire pour cet exemple
      return(iris)
    } else if (!is.null(input$file)) {
      # Charger un fichier utilisateur
      df <- read.csv(input$file$datapath)
      return(df)
    }
    return(NULL)
  })
  
  model <- reactiveVal(NULL)  # Stocker le modèle entraîné
  results <- reactiveValues(conf_matrix = NULL, loss_plot = NULL)  # Stocker les résultats
  
  observeEvent(input$train, {
    req(data())  # Vérifier que les données existent
    
    df <- data()
    target_col <- ncol(df)  # Supposons que la dernière colonne est la cible
    X <- as.matrix(df[, -target_col])
    y <- as.factor(df[, target_col])
    
    # Diviser les données en train/test
    set.seed(42)
    train_index <- createDataPartition(y, p = 0.7, list = FALSE)
    X_train <- X[train_index, ]
    y_train <- y[train_index]
    X_test <- X[-train_index, ]
    y_test <- y[-train_index]
    
    # Initialiser et entraîner le modèle
    logistic_model <- LogisticRegressionMultinomial$new(
      learning_rate = input$learning_rate, 
      num_iterations = input$num_iterations
    )
    logistic_model$fit(X_train, y_train)
    model(logistic_model)  # Stocker le modèle
    
    # Générer les prédictions
    predictions <- logistic_model$predict(X_test)
    confusion_matrix <- table(Predicted = predictions, Actual = y_test)
    results$conf_matrix <- confusion_matrix
    
    # Stocker la courbe de perte
    results$loss_plot <- logistic_model$loss_history
  })
  
  # Afficher le résumé
  output$summary <- renderPrint({
    req(model())
    logistic_model <- model()
    
    df <- data()
    target_col <- ncol(df)
    X <- as.matrix(df[, -target_col])
    y <- as.factor(df[, target_col])
    
    # Diviser les données en train/test
    set.seed(42)
    train_index <- createDataPartition(y, p = 0.7, list = FALSE)
    X_test <- X[-train_index, ]
    y_test <- y[-train_index]
    
    logistic_model$summary(X_test, y_test)
  })
  
  # Afficher la courbe de perte
  output$loss_plot <- renderPlot({
    req(results$loss_plot)
    plot(results$loss_plot, type = "l", col = "blue", lwd = 2,
         main = "Loss Function Convergence",
         xlab = "Iterations", ylab = "Loss")
  })
  
  # Afficher la matrice de confusion
  output$conf_matrix <- renderTable({
    req(results$conf_matrix)
    results$conf_matrix
  })
}

# Lancer l'application
shinyApp(ui = ui, server = server)
