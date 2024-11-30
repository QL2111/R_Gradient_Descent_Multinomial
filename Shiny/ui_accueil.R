# ui_accueil.R
library(shiny)

# Interface pour l'onglet Accueil
accueil_ui <- fluidPage(
  h2("Bienvenue !"),
  textOutput("accueil_text")
)
