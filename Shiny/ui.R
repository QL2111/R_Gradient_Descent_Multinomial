# ui.R
library(shiny)

# Charger les composants de chaque onglet
source("ui_accueil.R")
source("ui_donnees.R")
source("ui_analyse.R")

# Interface utilisateur principale avec un lien vers le fichier CSS
ui <- navbarPage(
  title = "Gradient Descent Multinomial",
  
  # Lien vers le fichier CSS dans le dossier www
  header = tags$head(
    includeCSS("www/style.css")
  ),
  
  # Définition des onglets
  tabPanel("Home", accueil_ui),
  tabPanel("Data", donnees_ui),
  tabPanel("Model", analyse_ui)
)
