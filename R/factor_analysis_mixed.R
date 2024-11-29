# factor_analysis_mixed.R

# PCA
# A refaire ??? -> Ne prends pas en compte les variables qualitatives, génère juste des valeurs aléatoires
# Il faudrait que factor_analysis_mixed prenne en compte les variables qualitatives de notre jeu de données
# Pour le moment on va utiliser le one-hot encoding car je ne suis pas sûr que factor_analysis_mixed soit fonctionnel -Quentin

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

