# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 09:26:57 2021

@author: GIANG Cécile
"""

#################### IMPORTATIONS LIBRAIRIES ET MODULES ####################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


################# CHARGEMENT DES DONNEES DE `POI-PARIS.PKL` #################

poidata = pickle.load(open("data/poi-paris.pkl","rb"))

## Choix d'un poi
typepoi = "clothing_store"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]), 2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]


########################### CODE SOURCE FOURNI ###########################


plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## Coordonnees GPS de la carte
xmin , xmax = 2.23,2.48
ymin , ymax = 48.806,48.916


def show_map():
    """ Affichage de la map de Paris.
        Notes: extent permet de controler l'echelle du plan
    """
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)


def show_poi(geo_mat):
    """ Affichage brut des poi.
        @param geo_mat: (float, float) array, liste des coordonnées des POIs
        Notes: alpha permet de regler la transparence, s la taille
    """
    plt.figure(figsize=(12,7))
    show_map()
    plt.scatter(geo_mat[:,1], geo_mat[:,0], alpha=0.8, s=3, color='blanchedalmond')

def show_density():
    """ Discretisation pour l'affichage des modeles d'estimation de densite.
    """
    steps = 100
    xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # A remplacer par res = monModele.predict(grid).reshape(steps,steps)
    res = np.random.random((steps,steps))
    plt.figure()
    show_map()
    plt.imshow(res, extent=[xmin,xmax,ymin,ymax], interpolation='none',\
                   alpha=0.3, origin = "lower")
    plt.colorbar()
    plt.scatter(geo_mat[:,0], geo_mat[:,1], alpha=0.3)


############ ESTIMATION DE DENSITE: METHODE DES HISTOGRAMMES #################

class Histogram_Model:
    """ Classe permettant de calculer la densité de probabilité des POIs
        dans la map par la méthode des histogrammes.
    """
    def __init__(self, geo_mat, bins):
        """ Constructeur de la classe Histogram_Model.
            @param geo_mat: (float, float) array, liste des coordonnées des POIs
            @param bins: float, pas de discrétisation
        """
        self.data = geo_mat
        self.bins = bins
        
        # Coordonnées de la map discrétisée et intervalle entre deux points
        self.xx, self.step = np.linspace(xmin, xmax, bins + 1, retstep = True)
        self.yy, self.step = np.linspace(ymin, ymax, bins + 1, retstep = True)
        
        # Effectif par case: bins-1 intervalles
        self.effectifs = np.zeros((bins, bins), dtype=float)
        self.densite = np.zeros((bins, bins), dtype=float)
    
    def train(self):
        """ Renvoie les effectifs des échantillons dans chaque case de la map.
        """
        # Remise à zero des effectifs et de la densité
        self.reset()
        
        # Calcul des effectifs des échantillons dans chaque case de la map
        for y, x in self.data:
            i = np.argmin(x >= self.xx) - 1
            j = np.argmin(y >= self.yy) - 1
            self.effectifs[j,i] += 1
        
        # Calcul de la densité à partir des effectifs
        self.densite = self.effectifs / np.sum(self.effectifs)
    
    def predict(self):
        """ Renvoie la densité des échantillons dans la map.
        """
        return self.densite
    
    def reset(self):
        """ Remise à zero des effectifs et de la densité.
        """
        self.effectifs = np.zeros((self.bins, self.bins), dtype=float)
        self.densite = np.zeros((self.bins, self.bins), dtype=float)
    

def show_density_Hist(geo_mat, bins):
    """ Affichage des densités calculées par la méthode des histogrammes.
        @param geo_mat: (float, float) array, liste des coordonnées des POIs
        @param bins: int, nombre d'intervalles par axes. Il y aura donc
                     bins x bines cases dans la grille. Plus bins est élevé,
                     plus le pas de discrétisation est petit.
    """
    plt.figure(figsize=(18,3))
    plt.title("Méthode des histogrammes, pas de %d" %(bins))
    
    # Calcul du modèle
    model = Histogram_Model(geo_mat, bins)
    model.train()
    density = model.predict()

    # Affichage
    show_map()
    plt.imshow(density,extent=[xmin,xmax,ymin,ymax],interpolation='none',alpha=0.3,aspect=1.5,origin ="lower",cmap='copper')
    plt.colorbar()
    plt.scatter(geo_mat[:,1], geo_mat[:,0], alpha=0.2, s=1, color='lavender')


############### ESTIMATION DE DENSITE: METHODE PAR NOYAUX ####################

class KDE_Model:
    """ Classe permettant de calculer la densité de probabilité des POIs
        dans la map par la méthode des noyaux (Kernel Density Estimation).
    """
    def __init__(self, geo_mat, h, noyau, steps):
        self.h = h
        self.noyau = noyau
        self.data = geo_mat
        
        # Calcul des points d'intérêt: ce sont tous le points de la grille grid
        self.steps = steps
        xx,yy = np.meshgrid(np.linspace(xmin,xmax,self.steps),np.linspace(ymin,ymax,self.steps))
        self.grid = np.c_[xx.ravel(), yy.ravel()]
        self.densite = np.zeros((len(self.grid)))
    
    def noyau_parzen(self, x, y):
        """ Fonction de Parzen.
            @param x: float, abscisse du point
            @param y: float, ordonnée du point
        """
        if np.linalg.norm(np.array([x, y])) <= 1/2:
            return 1
        else:
            return 0
    
    def noyau_gauss(self, x, y):
        """ Noyau de Gauss.
            @param x: float, abscisse du point
            @param y: float, ordonnée du point
        """
        return np.exp(- np.linalg.norm(np.array([x, y]))**2 / 2) / np.sqrt(2 * np.pi)
    
    def noyau_laplace(self, x, y):
        """ Noyau de Laplace.
            @param x: float, abscisse du point
            @param y: float, ordonnée du point
        """
        return 0.5 * np.exp(- np.linalg.norm(np.array([x, y])))
    
    def predict(self):
        # Calcul du nombre de points dans l'hypercube centré en chaque
        # point de grid
        if self.noyau == 'parzen':
            for i in range(len(self.densite)):
                # Point d'intérêt
                x0, y0 = self.grid[i]
                self.densite[i] = sum([self.noyau_parzen((x0-x)/self.h , (y0 - y)/self.h) / self.h**2 for y, x in self.data])
        if self.noyau == 'gauss':
            for i in range(len(self.densite)):
                # Point d'intérêt
                x0, y0 = self.grid[i]
                self.densite[i] = sum([self.noyau_gauss((x0-x)/self.h , (y0 - y)/self.h) / self.h**2 for y, x in self.data])
        if self.noyau == 'laplace':
            for i in range(len(self.densite)):
                # Point d'intérêt
                x0, y0 = self.grid[i]
                self.densite[i] = sum([self.noyau_laplace((x0-x)/self.h , (y0 - y)/self.h) / self.h**2 for y, x in self.data])
        
        self.densite = self.densite / len(self.data)
        return self.densite

def show_density_KDE(geo_mat, h, noyau, steps):
    """ Affichage des densités calculées par la méthode des histogrammes.
        @param step: nombre d'intervalles par axes de la grille
    """
    plt.figure(figsize=(18,3))
    title = "Méthode par noyau de " + noyau + ", h = " + str(h) + ", steps = " + str(steps)
    plt.title(title)
    
    # Calcul du modèle
    density = KDE_Model(geo_mat, h, noyau, steps).predict()
    
    # Affichage
    show_map()
    plt.imshow(density.reshape(steps,steps),extent=[xmin,xmax,ymin,ymax],interpolation='none',alpha=0.3,aspect=1.5,origin ="lower",cmap='copper')
    plt.colorbar()
    plt.scatter(geo_mat[:,1], geo_mat[:,0], alpha=0.2, s=1, color='lavender')
    

############### ESTIMATION DE DENSITE: METHODE PAR NOYAUX ####################

class KDE_Model:
    """ Classe permettant de calculer la densité de probabilité des POIs
        dans la map par la méthode des noyaux (Kernel Density Estimation).
    """
    def __init__(self, geo_mat, h, noyau, steps):
        self.h = h
        self.noyau = noyau
        self.data = geo_mat
        
        # Calcul des points d'intérêt: ce sont tous le points de la grille grid
        self.steps = steps
        xx,yy = np.meshgrid(np.linspace(xmin,xmax,self.steps),np.linspace(ymin,ymax,self.steps))
        self.grid = np.c_[xx.ravel(), yy.ravel()]
        self.densite = np.zeros((len(self.grid)))
    
    def noyau_parzen(self, x, y):
        """ Fonction de Parzen.
            @param x: float, abscisse du point
            @param y: float, ordonnée du point
        """
        if np.linalg.norm(np.array([x, y])) <= 1/2:
            return 1
        else:
            return 0
    
    def noyau_gauss(self, x, y):
        """ Noyau de Gauss.
            @param x: float, abscisse du point
            @param y: float, ordonnée du point
        """
        return np.exp(- np.linalg.norm(np.array([x, y]))**2 / 2) / np.sqrt(2 * np.pi)
    
    def noyau_laplace(self, x, y):
        """ Noyau de Laplace.
            @param x: float, abscisse du point
            @param y: float, ordonnée du point
        """
        return 0.5 * np.exp(- np.linalg.norm(np.array([x, y])))
    
    def predict(self):
        # Calcul du nombre de points dans l'hypercube centré en chaque
        # point de grid
        if self.noyau == 'parzen':
            for i in range(len(self.densite)):
                # Point d'intérêt
                x0, y0 = self.grid[i]
                self.densite[i] = sum([self.noyau_parzen((x0-x)/self.h , (y0 - y)/self.h) / self.h**2 for y, x in self.data])
        if self.noyau == 'gauss':
            for i in range(len(self.densite)):
                # Point d'intérêt
                x0, y0 = self.grid[i]
                self.densite[i] = sum([self.noyau_gauss((x0-x)/self.h , (y0 - y)/self.h) / self.h**2 for y, x in self.data])
        if self.noyau == 'laplace':
            for i in range(len(self.densite)):
                # Point d'intérêt
                x0, y0 = self.grid[i]
                self.densite[i] = sum([self.noyau_laplace((x0-x)/self.h , (y0 - y)/self.h) / self.h**2 for y, x in self.data])
        
        self.densite = self.densite / np.sum(self.densite)
        return self.densite


############ PREDICTION DES NOTES EN FONCTION DE LA LOCALISATION ############

class Nadaraya_Watson:
    
    def __init__(self, data_train, noyau, h):
        self.h = h
        self.noyau = noyau
        
        self.geo_mat = data_train[:,:2]
        self.notes = data_train[:,2]
        
    def noyau_parzen(self, x, y):
        return np.where((np.abs(x-self.geo_mat[:,0]) < self.h/2) & (np.abs(y-self.geo_mat[:,1]) < self.h/2), 1, 0)
    
    def noyau_gauss(self, x, y):
        return np.exp(-0.5*((((self.geo_mat[:,0]-x) / self.h)**2) + ((self.geo_mat[:,1]-y) / self.h)**2)) / (np.sqrt(2*np.pi) * self.h)

    def predict(self, data_test):
        
        # Notes prédites
        notes_pred = np.zeros(len(data_test))
        
        for i, x in enumerate(data_test):
            # Cas noyau Parzen
            if self.noyau == 'parzen':
                ponderations = self.noyau_parzen(x[0], x[1])
               
            # Cas noyau de Gauss
            elif self.noyau == 'gauss':
                ponderations = self.noyau_gauss(x[0], x[1])
                
            notes_pred[i] = np.sum(self.notes*ponderations) / np.sum(ponderations)
            
        return notes_pred


class KNN:
    
    def __init__(self, data_train, k):
        self.k = k
        self.geo_mat = data_train[: , :2]
        self.notes = data_train[: , 2]
        
    def predict(self, data_test):
        
        notes_pred = np.zeros(len(data_test))
        
        for i, x in enumerate(data_test):
            norm = np.linalg.norm(self.geo_mat - x[:2], axis=1)
            voisins = self.notes[np.argsort(norm)[ : self.k]]
            notes_pred[i] = np.mean(voisins)
            
        return notes_pred
