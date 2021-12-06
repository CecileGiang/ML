# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 08:36:22 2021

@author: GIANG Cécile
"""

################### IMPORTATION DES LIBRAIRIES ET MODULES ###################

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas

from mltools import plot_data, plot_frontiere, make_grid, gen_arti


#################### IMPLEMENTATION DES FONCTIONS DE COÛT ###################

def mse(w, x, y):
    """ Renvoie le coût aux moindres carrés pour une fonction linéaire de 
        paramètres w de taille (d) sur les données x de taille (n,d) et les 
        labels y (n).
    """
    # --- Reformatage des entrées pour éviter les bugs
    
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    #x = x.reshape(y.shape[0],w.shape[0])
    
    return (y - np.dot(x,w))**2

def reglog(w, x, y):
    """ Renvoie le coût pour une régression logistique sur des paramètre w de
        taille (d), x de taille (n,d) et y (n).
    """
    # --- Reformatage des entrées pour éviter les bugs
    
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    #x = x.reshape(y.shape[0],w.shape[0])
    
    fw = np.dot(x,w)
    return np.log( 1 + np.exp( -y * fw) )

def mse_grad(w, x, y):
    """ Renvoie le gradient des moindres carrés sous la forme d'une 
        matrice (n,d).
    """
    # --- Reformatage des entrées pour éviter les bugs
    
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    #x = x.reshape(y.shape[0],w.shape[0])
    
    return -2 * x * (y - np.dot(x,w))

def reglog_grad(w, x, y):
    """ Renvoie le gradient de la régression logistique sous la forme d'une
        matrice (n,d).
    """
    # --- Reformatage des entrées pour éviter les bugs
    
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    #x = x.reshape(y.shape[0],w.shape[0])
    
    fw = np.dot(x,w)
    return (- y * x ) / ( 1 + np.exp( y * fw ) )

def grad_check(f, f_grad, eps = 0.05, N = 100):
    """ Tire au hasard N points de dimension 1 et vérifie sur ces N points 
        le calcul du gradient.
        @return : float, moyenne des différences de gradient pour les 2 méthodes
        @return : pd.DataFrame, DataFrame des différences, sert pour l'affichage
    """
    # --- Tirage aléatoire des exemples de X, w et y
    
    x = np.array( [rd.randint(0,100) for i in range(N)] ).reshape(-1,1)
    w = np.array([rd.randint(1,10)])
    y = np.array(rd.choice([-1,1]))
    
    # Dictionnaire servant à stocker les différentes valeurs de gradient (pour f et Taylor)
    grad_values = dict()
    
    # diff : moyenne des différences entre les 1 méthodes de calcul du gradient
    diff = 0
    
    for i in range(N):
        # Calcul des gradients par f et par DL de Taylor
        grad_approx = f_grad(w + eps, x[i], y)
        grad_dl = ( f( w + eps, x[i], y ) - f( w, x[i], y ) ) / eps
        
        # Mise à jour du dictionnaire et de la moyenne des différences
        grad_values[i] = {f.__name__ : "%.1f" % grad_approx[0][0], 'Taylor' : "%.1f" % grad_dl[0][0]}
        diff += grad_approx - grad_dl
        
    return diff / N, pandas.DataFrame.from_dict(grad_values)


########################### DESCENTE DE GRADIENT ###########################

def descente_gradient(datax, datay, f_loss, f_grad, eps, maxIter, descent='batch', mb = 10):
    """ Réalise une descent de gradient pour optimier le coût f_loss (de 
        gradient f_grad) sur les données datax et les labels datay, avec un 
        pas de descente de eps et maxIter itérations.
        @return w: array, w optimal trouvé au bout de maxIter itérations
        @return allw: list(array), liste des w successivement calculés
        @return allf: list(float), liste des coût pour toutes les itérations
    """
    # Initialisation de w, allw, allf
    w = np.zeros((1, len(datax[0])))
    allw = [ w[0].tolist() ]                            # tous les w calculés
    allf = [ np.mean( f_loss( w, datax, datay ) ) ]     # tous les coûts
    
    for niter in range(maxIter):
        if descent == 'batch':
            # On moyenne les ∂f/∂wi pour chaque dimension i ∈ [1,d] 
            delta = np.mean( f_grad( w, datax, datay), axis = 0 )
            w -= eps * delta
        if descent == 'stochastique':
            # On tire au hasard un exemple de datax et on met à jour w
            i = rd.randint(0, len(datax)-1)
            x = datax[i]
            y = datay[i]
            w -= eps * f_grad( w, x, y)
        if descent == 'mini-batch':
            inds = [rd.randint(0,len(datax)-1) for i in range(mb)]
            delta = np.mean( [ f_grad( w, datax[i], datay[i]) for i in inds], axis = 0 )
            w -= eps * delta
            
        # Mise à jour de allw et allf
        allw.append( w[0].tolist() )
        allf.append(np.mean( f_loss( w, datax, datay ) ) )
        
    return w, allw, allf


class RegLineaire:
    """ Classe pour la régression linéaire, qui permet de trouver la valeur 
        du coefficient a pour un ensemble de points suivant ax + eps, où eps
        est un bruit.
    """
    def __init__(self, xtrain, ytrain):
        """ @param datax: float array, abscisses (1 dimension)
            @param datay: float array, ordonnées (1 dimension)
        """
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.w = None
        self.allw = None
        self.allf = None
        
    def fit(self, eps=0.01, maxIter=1000, descent = 'batch'):
        """ Phase d'entraînement pour retrouver le coefficient a.
            Nous procédons par descente de gradient afin de trouver le 
            paramètre w (correspond à a) qui minimise le coût aux moindres 
            carrés.
        """
        self.w, self.allw, self.allf = descente_gradient(self.xtrain, self.ytrain, mse, mse_grad, eps=eps, maxIter=maxIter, descent=descent, mb = 10)
    
    def predict(self, xtest, ytest):
        """ Phase de test.
            y = np.sign( w_1 * x_1 + w_2 * x_2 + ... + w_d * x_d )
        """
        self.xtest =  xtest
        self.ytest = ytest
        return np.array( [ np.sign( np.vdot( xtest[i], self.w) ) for i in range( len(xtest) ) ] )
    
    def score(self):
        """ Calcule les scores de prédiction sur les données d'entraînement 
            et les données de test.
        """
        # Taux de bonne classification sur les données train
        pred_train = self.predict( self.xtrain, self.ytrain )
        score_train = np.mean( [ 1 if self.ytrain[i] == pred_train[i] else 0 for i in range(len( self.ytrain )) ] )
        
        # Taux de bonne classification sur les données test
        pred_test = self.predict( self.xtest, self.ytest )
        score_test = np.mean( [ 1 if self.ytest[i] == pred_test[i] else 0 for i in range(len( self.ytest )) ] )
        
        return score_train, score_test
    
    def display(self):
        """ Affichage des données (points de datax et datay) et de la droite 
            prédite: wx.
        """
        toPlot = [ self.w[0][0] * x[0] for x in self.xtrain ]
        plt.figure()
        plt.plot('Régression linéaire')
        plt.scatter(self.xtrain.reshape(1,-1), self.ytrain, s = 1, c = 'midnightblue')
        plt.plot(self.xtrain.reshape(1,-1)[0], toPlot, color = 'mediumslateblue')
    
    def getw(self):
        """ Getteur du paramètre w optimal.
        """
        return self.w.reshape( self.xtrain.shape[1] , 1 )
    
    def getallw(self):
        """ Getteur du paramètre allw.
        """
        return self.allw
    
    def getallf(self):
        """ Getteur du paramètre allf.
        """
        return self.allf
    

class RegLogistique:
    """ Classe pour la régression logistique, qui permet de prédire la classe
        y ∈ {-1,1} d'échantillons x, après s'être entraîné sur les données de 
        xtrain et ytrain.
    """
    def __init__(self, xtrain, ytrain):
        """ @param xtrain: float array x array, base d'exemples
            @param ytrain: int array, liste des classes
        """
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = None
        self.ytest = None
        self.w = None
        self.allw = None
        self.allf = None
    
    def fit(self, eps=0.01, maxIter=1000, descent = 'batch'):
        """ Phase d'entraînement pour retrouver le coefficient w par descente 
            de gradient qui minimise le coût logistique (reglog).
        """
        self.w, self.allw, self.allf = descente_gradient(self.xtrain, self.ytrain, reglog, reglog_grad, eps=eps, maxIter=maxIter, descent=descent, mb = 10)
    
    def predict(self, xtest, ytest):
        """ Phase de tests. Applique aux données de test xtrain la fonction 
            fw(x), qui fait le produit scalaire entre w et chaque x de xtrain.
            Le seuil de prédiction est 0:
                * si fw(x) < 0: prédit -1
                * sinon: prédit 1
        """
        # On garde en mémoire les données de test
        self.xtest = xtest
        self.ytest = ytest
        
        # La classe d'un exemple x correspond au signe de fw(x)
        return np.array( [ np.sign( np.vdot( x, self.w ) ) for x in xtest ] )
    
    def score(self):
        """ Calcule les scores de prédiction sur les données d'entraînement 
            et les données de test.
        """
        # Taux de bonne classification sur les données train
        pred_train = self.predict( self.xtrain, self.ytrain )
        score_train = np.mean( [ 1 if self.ytrain[i] == pred_train[i] else 0 for i in range(len( self.ytrain )) ] )
        
        # Taux de bonne classification sur les données test
        pred_test = self.predict( self.xtest, self.ytest )
        score_test = np.mean( [ 1 if self.ytest[i] == pred_test[i] else 0 for i in range(len( self.ytest )) ] )
        
        return score_train, score_test
    
    def getw(self):
        """ Getteur du paramètre w optimal. On le reformate pour la 
            visualisation.
        """
        return self.w.reshape( self.xtrain.shape[1] , 1 )
    
    def getallw(self):
        """ Getteur du paramètre allw.
        """
        return self.allw
    
    def getallf(self):
        """ Getteur du paramètre allf.
        """
        return self.allf
    

##################### EXPERIMENTATIONS ET VISUALISATION #####################

def main( xtrain, ytrain, xtest, ytest, m = 'lineaire', eps= 0.01, maxIter=1000, descent = 'batch' ):
    """ Visualisation pour des données d'entraînement xtrain et ytrain.
        @param m: str, modèle de régression utilisé: 'lineaire' ou 'logistique'
                  @default-value: 'lineaire'
        @param eps: list(float), liste des pas pour la descente de gradient
        @param maxIter: int, nombre d'itération pour la descente de gradient
        @param descent: str, 'batch', 'stochastique' ou 'mini-batch'
                  @default-value: 'batch'
    """
    # Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
    
    fig = plt.figure(figsize=(20,4))
    plt.title('Régression %s pour eps = %f ' % (m,eps), y = 1.2)
        
    # Création du modèle
    rl = RegLineaire(xtrain, ytrain) if m == 'lineaire' else RegLogistique(xtrain, ytrain)
    rl.fit(eps, maxIter, descent)
    rl.predict(xtest, ytest)
    w = rl.getw().reshape(xtrain.shape[1],1)
        
    # Affichage des données et frontière de décision pour xtest
    ax1 = fig.add_subplot(141)
    ax1.title.set_text('Score train : %f' % rl.score()[0] )
    ax1 = plot_frontiere(xtrain, lambda x : np.sign(x.dot(w)),step=100)
    ax1 = plot_data(xtrain, ytrain.reshape(1,-1)[0])
    
    # Affichage des données et frontière de décision pour xtrain
    ax2 = fig.add_subplot(142)
    ax2.title.set_text('Score test : %f' % rl.score()[1] )
    ax2 = plot_frontiere(xtest, lambda x : np.sign(x.dot(w)),step=100)
    ax2 = plot_data(xtest, ytest.reshape(1,-1)[0])
        
    # Visualisation de la fonction de coût en 2D
    ax3 = fig.add_subplot(143)
    ax3.title.set_text('Fonction de coût %s' % m )
    allw = rl.getallw()
    if m == 'lineaire':
        ax3 = plt.contourf(x, y, np.array([ np.mean( mse(w, xtrain, ytrain) ) for w in grid]).reshape(x.shape),cmap ='BuPu',levels=50)
        plt.scatter( [w[0] for w in allw] , [w[1] for w in allw], c='lightsteelblue', marker='.')
    else:
        ax3 = plt.contourf(x, y, np.array([ np.mean( reglog(w, xtrain, ytrain) ) for w in grid]).reshape(x.shape), cmap ='BuPu',levels=50)
        plt.scatter( [w[0] for w in allw] , [w[1] for w in allw], cmap='lightsteelblue', marker='.')
        
    # Visualisation de l'évolution de la fonction de coût
    allf = rl.getallf()
    ax4 = fig.add_subplot(144)
    ax4.title.set_text('Evolution du coût au fil des itérations pour %s' % m )
    ax4 = plt.plot(allf, color = 'peru')