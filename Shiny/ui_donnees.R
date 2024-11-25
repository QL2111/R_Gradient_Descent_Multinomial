library(shiny)

# Interface pour l'onglet Données
donnees_ui <- fluidPage(
  h2("Page Données"),
  
  # Ajout des panneaux
  tabsetPanel(
    
    # Premier panneau : Importation et affichage des données
    tabPanel(
      title = "Importation",
      
      # Choix du fichier à importer (CSV ou Excel)
      fileInput("file", "Importer un fichier CSV ou Excel",
                accept = c(".csv", ".xls", ".xlsx")),
      
      # Choix du délimiteur pour les fichiers CSV
      conditionalPanel(
        condition = "input.file && input.file.name.endsWith('.csv')",
        selectInput("delimiter", "Sélectionnez le délimiteur",
                    choices = c("Virgule" = ",",
                                "Point-virgule" = ";",
                                "Tabulation" = "\t",
                                "Espace" = " "),
                    selected = ",")
      ),
      
      # Tableau pour afficher les données importées
      tableOutput("data_table")
    ),
    
    # Deuxième panneau : Texte explicatif
    tabPanel(
      title = "Exploration",
      
      # Sélecteur pour choisir une variable
      selectInput("var_to_plot", "Choisissez une variable",
                  choices = NULL,  # Les choix seront définis dynamiquement
                  selected = NULL),
      
      # Cadre pour afficher le graphique
      plotOutput("dynamic_plot", height = "400px")
      
    ),
    
    # Troisième panneau : Texte explicatif supplémentaire
    tabPanel(
      title = "Panneau 3",
      h3("Contenu du panneau 3"),
      p("Vous pouvez utiliser ce panneau pour des informations supplémentaires.")
    )
  )
)
