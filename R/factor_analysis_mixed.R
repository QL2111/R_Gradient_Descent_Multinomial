# factor_analysis_mixed.R

# PCA
# A refaire ??? -> Ne prends pas en compte les variables qualitatives, génère juste des valeurs aléatoires
# Il faudrait que factor_analysis_mixed prenne en compte les variables qualitatives de notre jeu de données
# Pour le moment on va utiliser le one-hot encoding car je ne suis pas sûr que factor_analysis_mixed soit fonctionnel -Quentin

factor_analysis_mixed <- function(data) {
  # Vérifier si les colonnes sont quantitatives ou qualitatives
  quantitative_vars <- sapply(data, is.numeric)
  qualitative_vars <- sapply(data, is.factor)
  
  if (!any(quantitative_vars) || !any(qualitative_vars)) {
    stop("Le jeu de données doit contenir à la fois des variables quantitatives et qualitatives.")
  }
  
  # 1. Standardisation des variables quantitatives
  data_quantitative <- scale(data[, quantitative_vars, drop = FALSE])
  
  # 2. Transformation des variables qualitatives avec one-hot encoding
  data_qualitative <- model.matrix(~ . - 1, data = data[, qualitative_vars, drop = FALSE])
  
  # Ajustement des poids pour les modalités (optionnel selon méthode souhaitée)
  # Diviser par la racine carrée du nombre de modalités de chaque variable
  modal_weights <- apply(data_qualitative, 2, sd)
  data_qualitative <- sweep(data_qualitative, 2, modal_weights, FUN = "/")
  
  # 3. Combinaison des deux types de données
  data_combined <- cbind(data_quantitative, data_qualitative)
  
  # 4. Analyse en composantes principales (ACP)
  covariance_matrix <- cov(data_combined)  # Matrice de covariance
  eig <- eigen(covariance_matrix)          # Décomposition en valeurs propres et vecteurs propres
  eigenvalues <- eig$values
  eigenvectors <- eig$vectors
  
  # Calcul des coordonnées des individus sur les composantes principales
  principal_components <- data_combined %*% eigenvectors
  
  # Retourner les premières dimensions principales
  coord <- as.data.frame(principal_components[, 1:2])  # Garder les deux premières dimensions
  colnames(coord) <- c("Dim1", "Dim2")
  
  return(coord)
}

