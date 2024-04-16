# recrrer_tensorflow

# Description de votre code de régression linéaire
Ce code Python implémente une régression linéaire personnalisée pour la prédiction. Voici une description détaillée de chaque partie :

# 1. Fonction erreur(X, y, params):

Calcule l'erreur quadratique moyenne (MSE) entre les valeurs prédites (P) et les valeurs réelles (y).
X: Matrice des features (observations x features).
y: Vecteur des valeurs cibles (observations x 1).
params: Dictionnaire contenant les poids (w) et le biais (b) du modèle.
Calculs principaux :
M = np.dot(X, params["w"]): Produit matriciel entre X et w pour obtenir les valeurs prédites linéaires (M).
P = M + params['b']: Ajout du biais (b) aux valeurs prédites.
L = np.mean((y-P)**2): Calcul du MSE en calculant la moyenne du carré des différences entre les valeurs réelles et les prédictions.
La fonction renvoie l'erreur (MSE) et un dictionnaire (info) contenant des informations intermédiaires utilisées plus tard.

# 2. Fonction gradian(info, param):

Calcule le gradient de l'erreur par rapport aux paramètres (w et b).
info: Dictionnaire contenant des informations issues de la fonction erreur.
param: Indique pour quel paramètre (w ou b) on veut calculer le gradient.

Calculs principaux :
Utilise le chaînage de règles pour calculer le gradient par rapport à w et b en utilisant les informations de info.
dl_dw et dl_db représentent les gradients pour w et b respectivement.
La fonction renvoie un dictionnaire (grad) contenant les gradients pour w et b.

# 3. Fonction train(X, y, epoch, learning_rate):

Entraîne le modèle de régression linéaire.
X: Matrice des features.
y: Vecteur des valeurs cibles.
epoch: Nombre d'itérations d'entraînement.
learning_rate: Taux d'apprentissage pour la mise à jour des poids.
Étapes principales :
Initialise les poids (w) et le biais (b) de façon aléatoire.
Crée une liste pour stocker les erreurs à chaque itération.
Boucle d'entraînement (epoch) :
Calcule l'erreur et les informations intermédiaires avec erreur(X, y, params).
Met à jour la liste des erreurs.
Calcule le gradient pour les poids et le biais avec gradian(info, params).
Met à jour les poids et le biais en utilisant la descente de gradient avec le taux d'apprentissage.
La fonction renvoie le dictionnaire des paramètres (params) et la liste des erreurs d'entraînement.

# 4. Fonction predict(X, params):

Effectue la prédiction sur de nouvelles données.
X: Matrice des features des nouvelles données.
params: Dictionnaire contenant les poids (w) et le biais (b) entraînés.
Calculs principaux :
Calcule les valeurs prédites en multipliant X par w et en ajoutant b.
La fonction renvoie un vecteur contenant les prédictions.

# 5. Fonctions d'évaluation mse, rmse, mae:

Ces fonctions permettent de calculer différentes mesures d'erreur :
## mse: Erreur quadratique moyenne (MSE).
## rmse: Racine carrée de l'erreur quadratique moyenne (RMSE).
## mae: Erreur absolue moyenne (MAE).
# 6. Entraînement et prédiction:
# 7. Analyse des résultats 
### Nous avions obtenir une prediction avec une erreur moyenne de 3.76
### comparer a une erreur de sklearne de 5.45
#### Notre fonctionne d'entrainement est plus ou moins proche de celle de sklearn


## Prochainement nous allons essayer de pousser loin en integrant une fonctions afin s'obtenir une regression non linéaire
# Tohouegnonjean
