# factor_analysis_mixed.R
# nolint start
# PCA

# A refaire ??? -> Ne prends pas en compte les variables qualitatives, génère juste des valeurs aléatoires
# Il faudrait que factor_analysis_mixed prenne en compte les variables qualitatives de notre jeu de données
# Pour le moment on va utiliser le one-hot encoding car je ne suis pas sûr que factor_analysis_mixed soit fonctionnel -Quentin

factor_analysis_mixed <- function(data) {
  # Vérifier si les colonnes sont quantitatives ou qualitatives
  quantitative_vars <- sapply(data, is.numeric)
  qualitative_vars <- sapply(data, is.factor)
  
  # Initialiser la liste pour stocker les données préparées
  prepared_list <- list()
  
  # 1. Standardiser les variables quantitatives
  if (any(quantitative_vars)) {
    quant_data <- data[, quantitative_vars, drop = FALSE] # drop=FALSE to keep data frame or else it will transform into a vector
    quant_data <- scale(data[, quantitative_vars, drop = FALSE])
    prepared_list$quantitative <- quant_data
  }
  
  # 2. Encodage des variables qualitatives
  if (any(qualitative_vars)) {
    qual_data <- data[, qualitative_vars, drop = FALSE]
    # One-hot encoding
    qual_data <- as.data.frame(model.matrix(~ . - 1, data = qual_data)) # ~ . - 1 means all columns except the first one (exclude intercept - to avoid multicollinearity)
    prepared_list$qualitative <- qual_data
  }
  
  # Ajustement des poids pour les modalités (optionnel selon méthode souhaitée)
  if (exists("prepared_list$qualitative")) {
    modal_weights <- apply(prepared_list$qualitative, 2, sd)
    prepared_list$qualitative <- sweep(prepared_list$qualitative, 2, modal_weights, FUN = "/")
  }
  
  # 3. Combinaison des deux types de données
  data_combined <- do.call(cbind, prepared_list)
  
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
# nolint end


# factor_analysis_mixed <- function(data) {
#   # Vérifier si les colonnes sont quantitatives ou qualitatives
#   quantitative_vars <- sapply(data, is.numeric)
#   qualitative_vars <- sapply(data, is.factor)
  
#   if (!any(quantitative_vars) || !any(qualitative_vars)) {
#     stop("Le jeu de données doit contenir à la fois des variables quantitatives et qualitatives.")
#   }
  
#   # 1. Standardisation des variables quantitatives
#   data_quantitative <- scale(data[, quantitative_vars, drop = FALSE])
  
#   # 2. Transformation des variables qualitatives avec one-hot encoding
#   data_qualitative <- model.matrix(~ . - 1, data = data[, qualitative_vars, drop = FALSE])
  
#   # Ajustement des poids pour les modalités (optionnel selon méthode souhaitée)
#   # Diviser par la racine carrée du nombre de modalités de chaque variable
#   modal_weights <- apply(data_qualitative, 2, sd)
#   data_qualitative <- sweep(data_qualitative, 2, modal_weights, FUN = "/")
  
#   # 3. Combinaison des deux types de données
#   data_combined <- cbind(data_quantitative, data_qualitative)
  
#   # 4. Analyse en composantes principales (ACP)
#   covariance_matrix <- cov(data_combined)  # Matrice de covariance
#   eig <- eigen(covariance_matrix)          # Décomposition en valeurs propres et vecteurs propres
#   eigenvalues <- eig$values
#   eigenvectors <- eig$vectors
  
#   # Calcul des coordonnées des individus sur les composantes principales
#   principal_components <- data_combined %*% eigenvectors
  
#   # Retourner les premières dimensions principales
#   coord <- as.data.frame(principal_components[, 1:2])  # Garder les deux premières dimensions
#   colnames(coord) <- c("Dim1", "Dim2")
  
#   return(coord)
# }