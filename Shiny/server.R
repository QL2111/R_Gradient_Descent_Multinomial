library(shiny)
library(DT)
library(Metrics)  # Pour calculer l'AUC
library(readxl)  # Pour lire les fichiers XLSX
library(pROC)
source("../R/DataPreparer.R")
library(R6)
source("../R/LogisticRegressionMultinomial.R")


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
  # observe({
  #   dataset <- data()
  #   if (!is.null(dataset)) {
  #     updateSelectInput(session, "target_var", choices = names(dataset), selected = names(dataset)[ncol(dataset)])
  #   }
  # })
  # Mettre à jour la sélection de la variable cible
  observe({
    dataset <- data()  # Récupérer les données actuelles
    if (!is.null(dataset)) {
      # Si le jeu de données est importé (par fichier), les colonnes de ce jeu sont affichées
      if (input$data_choice == "uploaded") {
        updateSelectInput(session, "target_var", choices = names(dataset), selected = names(dataset)[ncol(dataset)])
      } else {
        # Sinon, utilisez les noms de colonnes de l'iris (jeu de données par défaut)
        updateSelectInput(session, "target_var", choices = names(dataset), selected = names(dataset)[ncol(dataset)])
      }
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
  
  observeEvent(input$train, {
    req(data())  # Assure que data() n'est pas NULL
    dataset <- data()
    target_var <- input$target_var
    
    # S'assurer que la variable cible est un facteur
    dataset[[target_var]] <- as.factor(dataset[[target_var]])
    
    # Vérifier si les données contiennent des lignes
    if (nrow(dataset) == 0) {
      showNotification("Le dataset est vide.", type = "error")
      return(NULL)
    }
    
    # Vérification des valeurs manquantes
    if (any(is.na(dataset))) {
      showNotification("Les données contiennent des valeurs manquantes. Veuillez les traiter avant de continuer.", type = "error")
      return(NULL)
    }
    
    # Étape de préparation des données
    prepare_data_instance <- DataPreparer$new(
      use_factor_analysis = as.logical(input$use_factor_analysis)
    )
    
    # Appeler `prepare_data` directement
    prepared_data <- prepare_data_instance$prepare_data(
      data = dataset,
      target_col = target_var,
      split_ratio = input$split_ratio,  # Ajoutez cette entrée à l'interface si elle est nécessaire
      stratify = as.logical(input$stratify),                 # Vous pouvez ajouter cette option si besoin
      remove_outliers = input$remove_outliers,
      outlier_seuil = input$outlier_threshold
    )
    
    # Accéder aux données préparées
    X_train <- prepared_data$X_train
    X_test <- prepared_data$X_test
    y_train <- prepared_data$y_train
    y_test <- prepared_data$y_test
    
    # Convertir les données préparées en matrices
    X_train_matrix <- as.matrix(X_train)
    X_test_matrix <- as.matrix(X_test)
    
    # Convertir la variable cible en valeurs numériques
    y_train_numeric <- as.numeric(y_train)
    y_test_numeric <- as.numeric(y_test)
    
    # Entraînement du modèle de régression logistique multinomiale
    logistic_model <- LogisticRegressionMultinomial$new(
      learning_rate = input$learning_rate,
      num_iterations = input$num_iterations,
      optimizer = input$optimizer,
      regularization = input$regularization,
      patience = input$patience,
      use_early_stopping = TRUE
    )
    
    logistic_model$fit(X_train_matrix, y_train_numeric)
    model(logistic_model)  # Stocker le modèle pour les étapes suivantes
    
    # Afficher une notification de succès
    showNotification("Le modèle a été entraîné avec succès.", type = "message", duration = 5)  # Affichage de la notification
    
    
    # Prédire sur l'ensemble de test
    predictions <- model()$predict(X_test_matrix) 
    
    probabilites <- model()$predict_proba(X_test_matrix)
    print(probabilites)
    
    # Calculer et afficher l'accuracy
    accuracy <- sum(predictions == y_test_numeric) / length(y_test_numeric)
    results$accuracy <- accuracy
    
    # Afficher le résumé du modèle
    output$summary_output <- renderPrint({
      req(model())
      model()$summary()
    })
    
    # Afficher les métriques
    output$metrics_output <- renderPrint({
      req(results$accuracy)
      cat("Accuracy: ", results$accuracy, "\n")
    })
    
    # Afficher la courbe de perte
    output$loss_plot <- renderPlot({
      req(model())
      model()$plot_loss()  # Utilisation de la méthode plot_loss pour afficher la courbe de perte
    })
    
    # Afficher la courbe roc
    output$roc_plot <- renderPlot({
      req(model())
      model()$plot_auc(X_test_matrix, y_test_numeric, probabilites)  # Utilisation de la méthode plot_loss pour afficher la courbe de perte
    })
    
    # Afficher graphe var importance
    output$var_imp_plot <- renderPlot({
      req(model())
      model()$var_importance()  # Utilisation de la méthode plot_loss pour afficher la courbe de perte
    })

  }) 
  
  # Exporter le modèle en PMML lorsque l'utilisateur appuie sur "Exporter"
  observeEvent(input$exportpmml, {
    req(model())  # Assurer que le modèle a bien été entraîné
    
    # Créer un fichier temporaire pour PMML
    temp_pmml_path <- tempfile(pattern = "model_", fileext = ".pmml")
    
    tryCatch({
      # Appeler la méthode export_pmml pour exporter le modèle
      model()$export_pmml(temp_pmml_path)
      
      # Créer un lien pour permettre à l'utilisateur de télécharger le fichier PMML
      output$download_link <- renderUI({
        downloadButton("download_pmml", "Télécharger le modèle")
      })
      
      # Gestion du téléchargement
      output$download_pmml <- downloadHandler(
        filename = function() {
          paste("model_", Sys.Date(), ".pmml", sep = "")
        },
        content = function(file) {
          file.copy(temp_pmml_path, file)
        }
      )
      
      # Notification de succès
      showNotification("Le modèle a été exporté avec succès. Cliquez pour le télécharger.", type = "message")
    }, error = function(e) {
      showNotification(paste("Erreur lors de l'exportation du modèle:", e$message), type = "error")
    })
  })
  
}  
