#!/usr/bin/env python
# coding: utf-8

# In[16]:


def erreur(X, y, params):
    info = {}
    #params = {"w":w, "b":b}
    # X est le featurs (une matrice de taille (m,n))
    # y la sortie
    M = np.dot(X,params["w"]) # dans ce cas w doit être de taille (n,1)
    P = M + params['b']
    L =  np.mean((y-P)**2)
    
    info['X'] = X
    info['y'] = y
    info['M'] = M
    info['P'] = P
    
    return L, info


# In[17]:


def gradian(info, param):
    grad = {}
    dl_dp = -2 * (info['y'] - info['P'])
    dp_dm = 1
    dm_dw = info['X'].T
    #pour w
    dl_dw = np.dot(dm_dw, dl_dp ) * dp_dm    
    # pour b
    dl_db = np.sum(dl_dp*1)
    
    grad['w'] = dl_dw
    grad['b'] = dl_db
    
    return grad


# In[29]:


def train(X,y, epoch, learning_rate):
    # weight initialisation
    params = {}
    
    n_features = X.shape[1]
    np.random.seed(42)
    params["w"] = np.random.randn(n_features,1)
    params["b"] = np.random.randn(1,1)
    
    liste_erreur = []
    # forward
    for i in range(epoch):
        loss, info = erreur(X,y,params)
        liste_erreur.append(loss)
        print(f'Epoch {i+1}............................ loss => {loss}')
        
        # backword 
        # le gradiant descent
        grad = gradian(info, params)
        
        #update
        for p in params:
            params[p] = params[p] - learning_rate * grad[p]
    return params, liste_erreur


# In[30]:


def predict(X, params):
    M = np.dot(X, params["w"])
    
    P = M + params['b']
    return P


# In[31]:


def mse(y, pred):
    return np.mean((y-pred)**2)

def rmse(y, pred):
    return np.sqrt(np.mean((y-pred)**2))

def mae(y, pred):
    return np.mean(np.abs(y-pred))


# # Entrainement du modele

# In[32]:


import pandas as pd
import numpy as np
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
data = data
target = target


# In[33]:


# redimentionnent
X = data
y = target.reshape(506,1)


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[40]:


params,erreurs = train(X_train, y_train, epoch=50, learning_rate=0.0001) 


# In[41]:


import matplotlib.pyplot as plt
plt.plot(erreurs)


# # Prédiction

# In[42]:


prediction = predict(X_test, params)

score_rmse = rmse(y_test, prediction)
score_mae = mae(y_test, prediction)


# In[43]:


print('RMse: {}'.format(score_rmse))
print('Mae: {}'.format(score_mae))


# # Comparaison avec la fonction sklearn

# In[44]:


from sklearn.linear_model import LinearRegression
modeles = LinearRegression()
modeles.fit(X_train, y_train)


prediction = modeles.predict(X_test)
score_rmse = rmse(y_test, prediction)
score_mae = mae(y_test, prediction)

print('RMse: {}'.format(score_rmse))
print('Mae: {}'.format(score_mae))


# # Nous avions obtenir une prediction avec une erreur moyenne de 3.76
# 
# ## comparer a une erreur de sklearne de 5.45
# 
# ## Notre fonctionne d'entrainement est plus ou moins proche de celle de sklearn

# # Merci 
