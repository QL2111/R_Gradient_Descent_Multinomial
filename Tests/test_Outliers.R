# Tester le pré-traitement avec titanic
# nolint start
train = read.csv("data/titanic_train.csv")
test = read.csv("data/titanic_test.csv")

# Afficher le nombre de valeurs manquantes dans chaque colonne
# print("Valeurs manquantes dans le train :")
# print(colSums(is.na(train)))
# print("Valeurs manquantes dans le test :")
# print(colSums(is.na(test)))

# Fonction pour détecter les outliers
detect_outliers <- function(x, seuil = 0.25) {
  if (is.numeric(x)) {
    Q1 <- quantile(x, seuil, na.rm = TRUE)
    Q3 <- quantile(x, 1 - seuil, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    return(sum(x < lower_bound | x > upper_bound, na.rm = TRUE))
  } else {
    return(NA)
  }
}

# Remove outliers
# Fonction pour supprimer les outliers
remove_outliers <- function(data, seuil = 0.25) {
  outlier_indices <- unique(unlist(sapply(data, detect_outliers, seuil = seuil)))
  outlier_indices <- outlier_indices[!is.na(outlier_indices)]  # Supprimer les valeurs NA
  if (length(outlier_indices) > 0) {
    return(data[-outlier_indices, ])
  } else {
    return(data)
  }
}



# Détecter les outliers dans les données d'entraînement
train_outliers <- sapply(train, detect_outliers, seuil = 0.10)
print("Outliers dans les données d'entraînement :")
print(train_outliers)
print(dim(train))

# Détecter les outliers dans les données de test
test_outliers <- sapply(test, detect_outliers, seuil = 0.10)
print("Outliers dans les données de test :")
print(test_outliers)
print(dim(test))

# Supprimer les outliers des données d'entraînement
train_clean <- remove_outliers(train, seuil = 0.10)
print("Données d'entraînement après suppression des outliers :")
print(dim(train_clean))

# Supprimer les outliers des données de test
test_clean <- remove_outliers(test, seuil = 0.10)
print("Données de test après suppression des outliers :")
print(dim(test_clean))

# nolint end
