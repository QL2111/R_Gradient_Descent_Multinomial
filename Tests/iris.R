# Charger le jeu de données Iris
data(iris)

# Le jeu Iris dataset contient la variable cible Species qui est une variable qualitative à 3 modalités, on pourra donc tester notre descente de gradient
# Préparer les données
X <- iris[, -5]                  # Variables prédictives (séparées)
y <- as.numeric(iris$Species)    # Variable cible transformée en numeric pour le modèle multinomial

# Conversion de y en une matrice de classe "one-hot" pour les calculs de gradient
y_one_hot <- model.matrix(~ y + 0)
print(head(y_one_hot))
print(X[1:5,])