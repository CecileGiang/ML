# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 08:41:57 2021

@author: GIANG Cécile
"""

################### IMPORTATION DES LIBRAIRIES ET MODULES ###################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random as rd
import itertools as it

from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from tme3 import *

    
############ IMPLEMENTATION DE LA FONCTION DE COUT DU PERCEPTRON ############

def perceptron_loss(w, x, y):
    """ Renvoie le coût perceptron max(0, -y <x,w>) sous la forme d'une 
        matrice (d), pour des données x de taille (n,d), des labels y de taille 
        (n) et un paramètre w de taille (d).
    """
    # --- Reformatage des entrées pour éviter les bugs
    
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    
    return np.maximum( 0 , -y * np.dot(x,w) )

def perceptron_grad(w, x, y):
    """ Renvoie le gradient du perceptron sous la forme d'une matrice (n,d).
    """
    # --- Reformatage des entrées pour éviter les bugs
    
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    
    # On cherche les points mal classés
    yx_w = ( y * np.dot(x,w) ).flatten()
    index_loss = np.where( yx_w < 0 )
    
    # Fonction de coût: pour un point xi, 0 si bien classé, -yi * xi sinon
    gradient = np.zeros(x.shape)
    gradient[index_loss] = (-y * x)[index_loss]
    
    return gradient


############### CLASSE LINEAIRE POUR LE PERCEPTRON LINEAIRE #################

class Lineaire:
    """ Classe pour le perceptron linéaire, qui permet de prédire la classe
        y ∈ {-1,1} d'échantillons x, après s'être entraîné sur les données de 
        xtrain et ytrain.
    """
    def __init__(self, loss = perceptron_loss, loss_g = perceptron_grad, proj = None, max_iter = 1000, eps = 0.01):
        """ @param loss: function, fonction de coût
            @param loss_g: function, fonction gradient correspondant
            @param max_iter: int, nombre maximal d'itérations pour la descente de gradient
            @param eps: float, pas de gradient
        """
        self.loss, self.loss_g = loss, loss_g
        self.proj = proj
        self.max_iter, self.eps = max_iter, eps
        
        self.w = None           # Vecteur poids
        self.allw = None        # Historique de tous les poids calculés par descente de gradient
        self.allf = None        # Historique des coûts obtenus à chaque itération de la descente de gradient
        
    def fit(self, xtrain, ytrain, descent = 'batch'):
        """ Phase d'entraînement pour retrouver le coefficient w par descente 
            de gradient qui minimise le coût du perceptron (perceptron_loss).
            Nous procédons par descente de gradient pendant max_iter itérations
            avec un pas eps en utilisant le coût loss et le gradient loss_g.
            @param datax: float array x array, base d'exemples d'entraînement
            @param datay: int array, liste des classes correspondant
            @param descent: str, type de descente effectué. Valeurs possibles:
                            'batch', 'stochastique', 'mini-batch'
        """        
        # Si besoin est, on projète avant de faire l'apprentissage
        if self.proj != None:
            self.xtrain = self.proj( xtrain )
        else:
            self.xtrain = xtrain
        self.ytrain = ytrain
            
        self.w, self.allw, self.allf = descente_gradient(self.xtrain, self.ytrain, self.loss, self.loss_g, eps=self.eps, maxIter=self.max_iter, descent=descent, mb = 10)

    def predict(self, xtest):
        """ Phase de tests. Prédit la classe de chaque x de datax en leur
            appliquant la fonction fw(x), qui fait le produit scalaire entre w 
            et chaque x de xtrain. Le seuil de prédiction est 0:
                * si fw(x) < 0: prédit -1
                * sinon: prédit 1
        """        
        if self.proj != None:
            self.xtest = self.proj( xtest )
        else:
            self.xtest = xtest
        
        return np.sign( np.dot( self.xtest, self.w.reshape(-1,1) ) )

    def score(self, xtest, ytest):
        """ Calcule les scores de prédiction sur les données d'entraînement 
            passées en paramètres.
        """
        self.xtest = xtest
        self.ytest = ytest
        
        # Taux de bonne classification sur les données test
        pred_test = self.predict( self.xtest )
        score_test = np.mean( [ 1 if self.ytest[i] == pred_test[i] else 0 for i in range(len( self.ytest )) ] )
        
        return score_test
    
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


######################## PROJECTION ET PENALISATION #########################

def proj_poly(datax):
    """ Renvoie la projection polynomiale de degré 2 des données de datax.
        @param datax: float array x array, données à projeter
    """
    proj = []
    for i in range(len(datax)):
        x = datax[i]
        tmp = [1]
        for xi in x:
            tmp += np.dot(xi, x).tolist()
        proj.append(tmp)
        
    return np.array(proj)

def proj_biais(datax):
    """ Permet de rajouter un biais à datax/
        @param datax: float array x array, données à projeter
    """
    return np.hstack( ( np.ones((datax.shape[0], 1)), datax ) )

def proj_gauss(datax, base, sigma):
    """ Renvoie la projection gaussienne des données de datax.
        @param datax: float array x array, données à projeter
        @param base: list(int), base
        @param sigma: float, écart-type
    """
    proj = []
    for i in range(len(datax)):
        x = datax[i]
        tmp = [1]
        for xi in x:
            tmp += np.dot(xi, x).tolist()
        proj.append(tmp)
        
    return np.array(proj)

def proj_gauss(datax, base=10,sigma=1,K=1):
    # Création d'une grille 2D
    x_max = np.max(datax)
    x_min = np.min(datax)
    x = np.linspace(x_min,x_max,base)
    pij = np.array(list(it.product(x, x)))
    
    # Initialisation
    s = np.zeros((len(datax),base**2))
    
    # Projection pour chaque exemple
    for i in range(len(datax)):
        # Vecteur de taille grid_size**2
        n = np.linalg.norm(datax[i]-pij,ord=2,axis=1) 
        s[i] = K*np.exp(-(n**2)/sigma)
    return s
    


################### CHARGEMENT ET TRAITEMENT DES DONNEES ####################

def load_usps(filename):
    """ Fonction de chargement des données.
        @param filename: str, chemin vers le fichier à lire
        @return datax: float array x array, données
        @return datay: float array, labels
    """
    with open(filename, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l, datax, datay):
    """ Fonction permettant de ne garder que 2 classes dans datax et datay.
        @param l: list(int), liste contenant les 2 classes à garder
        @param datax: float array x array, données
        @param datay: float array, labels
        @param datax_new: float array x array, données pour 2 classes
        @param datay_new: float array, labels pour 2 classes
    """
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    datax_new, datay_new = np.vstack(tmp[0]),np.hstack(tmp[1])
    
    return datax_new, datay_new

def show_usps(datax):
    """ Fonction d'affichage des données.
        
    """
    plt.imshow(datax.reshape((16,16)),interpolation="nearest",cmap="gray")



##################### EXPERIMENTATIONS ET VISUALISATION ######################

def split_data_usps(neg, pos):
    """ Chargement des données USPS et isolation de deux classes neg et pos.
        @param neg: int, première classe à isoler
        @param pos: int, deuxième classe à isoler
    """
    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    xtrain,ytrain = get_usps([neg,pos],alltrainx,alltrainy)
    xtest,ytest = get_usps([neg,pos],alltestx,alltesty)
    
    
    # On remet les labels de la classe neg à -1 et ceux de la classe pos à 1
    ytrain = np.where(ytrain==neg, -1, ytrain)
    ytrain = np.where(ytrain==pos, 1, ytrain)
    ytest = np.where(ytest==neg, -1, ytest)
    ytest = np.where(ytest==pos, 1, ytest)
    
    return xtrain, ytrain, xtest, ytest

def shuffle_data(datax, datay):
    """ Pour mélanger un jeu de données.
    """
    temp = list(zip(datax, datay)) 
    np.random.shuffle(temp) 
    res1, res2 = zip(*temp)
    
    return np.array(res1), np.array(res2)

def showError( xtrain, ytrain, xtest, ytest, eps= 0.01, maxIter=1000, descent = 'batch' ):
    """ Application des fonctions ci-dessus sur les données USPS.
        Visualisation pour des données d'entraînement xtrain et ytrain.
        @param eps: list(float), liste des pas pour la descente de gradient
        @param maxIter: int, nombre d'itération max pour la descente de gradient
        @param descent: str, 'batch', 'stochastique' ou 'mini-batch'
                  @default-value: 'batch'
    """
    # On prendra les niter par pas de 100
    iters = np.arange(0, maxIter + 1, 100)
    
    # Liste des taux de mauvaise classification
    err_train = []
    err_test = []
    
    for niter in iters:
        # Création du modèle
        model = Lineaire(loss = perceptron_loss, loss_g = perceptron_grad, max_iter = niter, eps = eps)
        model.fit(xtrain, ytrain, descent = descent)
        err_train.append( 1 - model.score(xtrain,ytrain) )
        err_test.append( 1 - model.score(xtest,ytest) )
    
    # Affichage de l'évolution de l'erreur
    fig = plt.figure(figsize=(12,2))
    plt.title('Evolution de l\'erreur au fil des itérations', y = 1.2)
    
    # Erreur en train
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Erreur apprentissage')
    ax1.set_ylim(0.,1.)
    ax1 = plt.plot(iters, err_train)
    
    # Erreur en test
    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Erreur test')
    ax2.set_ylim(0.,1.)
    ax2 = plt.plot(iters, err_test)


def main( xtrain, ytrain, xtest, ytest, proj = proj_poly, eps= 0.01, maxIter=1000, descent = 'batch' ):
    """ Visualisation pour des données d'entraînement xtrain et ytrain.
        @param eps: list(float), liste des pas pour la descente de gradient
        @param maxIter: int, nombre d'itération pour la descente de gradient
        @param descent: str, 'batch', 'stochastique' ou 'mini-batch'
                  @default-value: 'batch'
    """
    # Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
    
    fig = plt.figure(figsize=(20,4))
    plt.title('Modèle perceptron pour eps = %f ' % eps, y = 1.2)
        
    # Création du modèle
    model = Lineaire(loss=perceptron_loss, loss_g=perceptron_grad, proj=proj, max_iter = maxIter, eps = eps)
    model.fit(xtrain, ytrain, descent = descent)
    w = model.getw() #.reshape(xtrain.shape[1],1)
        
    # Affichage des données et frontière de décision pour xtrain
    ax1 = fig.add_subplot(131)
    ax1.title.set_text('Score train : %f' % model.score(xtrain, ytrain) )
    ax1 = plot_frontiere(xtrain, model.predict, step=50)
    ax1 = plot_data(xtrain, ytrain.reshape(1,-1)[0])
    
    # Affichage des données et frontière de décision pour xtest
    ax2 = fig.add_subplot(132)
    ax2.title.set_text('Score test : %f' % model.score(xtest, ytest) )
    ax2 = plot_frontiere(xtest, model.predict,step=50)
    ax2 = plot_data(xtest, ytest.reshape(1,-1)[0])
        
    # Visualisation de l'évolution de la fonction de coût
    allf = model.getallf()
    ax3 = fig.add_subplot(133)
    ax3.title.set_text('Evolution du coût au fil des itérations')
    ax3 = plt.plot(allf, color = 'peru')