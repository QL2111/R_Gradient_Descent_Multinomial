# factor_analysis_mixed.R
# nolint start
# PCA

# Chargement des packages nécessaires
library(FactoMineR)
library(factoextra)

# Fonction factor_analysis_mixed
factor_analysis_mixed <- function(data) {
  # Vérification des types de colonnes
  if (!is.data.frame(data)) {
    stop("Le jeu de données doit être une data.frame.")
  }
  # Application de l'Analyse Factorielle des Données Mixtes (FAMD)
  famd_result <- FAMD(data, graph = FALSE, ncp = 6)
  return(famd_result)
}
# nolint end