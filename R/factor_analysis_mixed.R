# factor_analysis_mixed.R


# A refaire ??? -> Ne prends pas en compte les variables qualitatives, génère juste des valeurs aléatoires
# Il faudrait que factor_analysis_mixed prenne en compte les variables qualitatives de notre jeu de données
# Pour le moment on va utiliser le one-hot encoding car je ne suis pas sûr que factor_analysis_mixed soit fonctionnel -Quentin

factor_analysis_mixed <- function(data) {
  return(data.frame(matrix(rnorm(n = nrow(data) * 2), ncol = 2)))
}
