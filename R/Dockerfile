# Utiliser une image de base R
FROM rocker/r-ver

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libxml2-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*s

# Copier les fichiers de votre projet dans le conteneur
COPY ./LogisticRegressionMultinomial.R /app/
COPY ./DataPreparer.R /app/

# Installer les bibliothèques R nécessaires
RUN Rscript -e "install.packages(c('glmnet', 'caret', 'R6', 'data.table', 'dplyr', 'ggplot2'))"

# Définir le répertoire de travail
WORKDIR /app

# Définir la commande par défaut
CMD ["Rscript", "LogisticRegressionMultinomial.R", "DataPreparer.R"]