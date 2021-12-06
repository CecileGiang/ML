# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:24:31 2021

@author: GIANG Cécile
"""

################### IMPORTATION DES LIBRAIRIES ET MODULES ####################

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import *
from sklearn.model_selection import cross_val_score
import itertools
import string
import unicodedata

from mltools import *
from tme3 import *
from tme4 import *


###################### PARSING DES FICHIERS DE DONNEES #######################

def load_mnist(filepath):
    """ Fonction de chargement des données MNIST (csv file).
        @param filepath: str, chemin vers le fichier .csv
        @return datax: (int) array x array, ensemble de données
        @return datay: (int) array, labels associés aux données
    """
    datax = []
    datay = []
    
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            datay.append(row[0])
            datax.append(row[1:])
            
    return np.array(datax, dtype=int) , np.array(datay, dtype=int)

def get_mnist(l, datax, datay):
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

def show_mnist(datax):
    """ Fonction d'affichage des données mnist.
        @param datax: (int) array, un échantillon de données (une chiffre)
    """
    plt.imshow(datax.reshape((28,28)), interpolation='nearest', cmap='gray')
    

############### LINEAIRE PENALISE - REGULARISATION DE TIKHONOV ###############

def ridge_loss(w, x, y, lamb = 1):
    """ Renvoie le coût perceptron max(0, -y <x,w>) régularisé sous la forme 
        d'une matrice (d), pour des données x de taille (n,d), des labels y 
        de taille (n) et un paramètre w de taille (d).
        Le paramètre lamb correspond à la pénalité associée au vecteur poids w.
    """
    # --- Reformatage des entrées pour éviter les bugs
    
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    
    return np.maximum( 0 , -y * np.dot(x,w) ) + lamb * np.sum(w**2)

def ridge_grad(w, x, y, lamb = 1):
    """ Renvoie le gradient du perceptron régularisé sous la forme d'une 
        matrice (n,d).
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
    
    return gradient + 2 * lamb * w.reshape(1,-1)


################ CLASSE LINREG POUR LE PERCEPTRON REGULARISE #################

class LinReg:
    """ Classe pour le perceptron régularisé, qui permet de prédire la classe
        y ∈ {-1,1} d'échantillons x, après s'être entraîné sur les données de 
        xtrain et ytrain.
    """
    def __init__(self, proj = None, max_iter = 1000, eps = 0.01):
        """ @param loss: function, fonction de coût
            @param loss_g: function, fonction gradient correspondant
            @param max_iter: int, nombre maximal d'itérations pour la descente de gradient
            @param eps: float, pas de gradient
        """
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
            
        self.w, self.allw, self.allf = descente_gradient(self.xtrain, self.ytrain, ridge_loss, ridge_grad, eps=self.eps, maxIter=self.max_iter, descent=descent, mb = 10)

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


############################# SVM ET GRIDSEARCH ##############################
    
class GridSearch:
    """ Classe pour trouver le meilleur noyau parmi ceux proposés dans l'
        attribut self.kernels, ainsi que le meilleur paramètre gamma, et le 
        meilleur paramètre degree dans le cas du noyau polynomial.
    """
    def __init__(self, kernels):
        """ Construction de la classe GridSearch.
            @param kernels: list(str), liste de noyaux
        """
        self.kernels = kernels
        self.degrees = [ d for d in range(1,10) ]
        self.gammas = {'scale', 'auto'}
    
    def test_model(self, model, xtrain, ytrain, xtest, ytest):
        """ Cross validation sur le modèle, renvoie le score en test moyen
        """
        model.fit(xtrain, ytrain)
        score_train = cross_val_score(model, xtrain, ytrain, cv=10)
        score_test = cross_val_score(model, xtest, ytest, cv=10)
        print('\tScore en train:', score_train)
        print('\tScore en test:', score_test)
        print('\tNombre de vecteurs supports:', model.n_support_)
        
        return np.mean(score_test)
    
    def score(self, xtrain, ytrain, xtest, ytest):
        """ Calcul des scores pour chaque modèle correspondant à un noyau.
            Estime le score par validation croisée (méthode `cross_val_score`
            du module `sklearn.model_selection`).
        """
        # Initialisation d'un dictionnaire regroupant les scores de test moyens
        # Indexation par les paramètres, les valeurs sont les scores
        scores = dict()
        
        for kernel in self.kernels:
            if kernel in ['rbf', 'poly', 'sigmoid']:
                for gamma in self.gammas:
                    if kernel == 'poly':
                        for degree in self.degrees:
                            print('\n----- Modèle %s pour gamma = %s, degree = %d' % (kernel, gamma, degree))
                            model = SVC( kernel = kernel, degree = degree, gamma = gamma )
                            scores[ kernel + ', ' + gamma + ', ' + str(degree) ] = self.test_model(model, xtrain, ytrain, xtest, ytest)
                    else:
                        print('\n----- Modèle %s pour gamma = %s' % (kernel, gamma))
                        model = SVC( kernel = kernel, gamma = gamma )
                        scores[ kernel + ', ' + gamma ] = self.test_model(model, xtrain, ytrain, xtest, ytest)
            else:
                print('\n----- Modèle %s' % kernel)
                model = SVC( kernel = kernel )
                scores[ kernel ] = self.test_model(model, xtrain, ytrain, xtest, ytest)
        
        index_best = list( scores.values() ).index( max(scores.values()) )
        
        print('\n\n---------- Meilleurs paramètres :', list( scores.keys() )[index_best] )
            
        return scores


######################### APPRENTISSAGE MULTI-CLASSE #########################

class OneVSOne:
    """ Classification multiclasse à partir de classifieurs binaires, méthode
        one vs one.
    """
    def __init__(self, kernel = 'rbf'):
        self.kernel = kernel
    
    def fit(self, xtrain, ytrain):
        """ Phase d'apprentissage. Apprend un classifieur permettant de 
            départager tout couple de labels.
        """
        # Liste de tous les labels du jeu de données
        self.labels = sorted( np.unique( ytrain ) )
        
        # Initialisation de la grille des classifieurs
        self.grid = [[ None for i in range(len(self.labels))] for j in range(len(self.labels))]
        
        for i in range(len(self.labels)):
            neg = self.labels[i]
            for j in range(len(self.labels)):
                pos = self.labels[j]
                # On ne distingue pas un label de lui-même
                if neg==pos: continue
                # Dans le cas où l'on peut déjà départager pos et neg...
                if neg > pos:
                    self.grid[neg][pos] = self.grid[pos][neg]
                # Sinon, on crée un classifieur neg vs pos
                else:
                    # Isolation des données neg et pos
                    datax, datay = get_usps([neg,pos],xtrain,ytrain)
                                
                    # On remet les labels de la classe neg à -1 et ceux de la classe pos à 1
                    datay = np.where(datay==neg, -1, datay)
                    datay = np.where(datay==pos, 1, datay)
                    
                    # Mélange des données
                    datax, datay = shuffle_data(datax, datay)
                    
                    # On rajoute à la grille le nouveau classifieur entraîné
                    self.grid[neg][pos] = SVC(kernel=self.kernel, gamma='auto', probability=True).fit(datax, datay)
    
    def predict(self, xtest):
        """ Phase de test. Le modèle va chercher la classe qui maximise les
            scores.
        """
        # Dictionnaire des classes prédites pour chaque modèle de self.grid
        classes = { i : [] for i in range(len(xtest)) }
        
        for i in range(len(self.grid)):
            for j in range(i, len(self.grid)):
                if i != j:
                    model = self.grid[i][j]
                    scores = model.predict_proba(xtest)
                    
                    preds = [ self.labels[i] if s[0] > s[1] else self.labels[j] for s in scores ]
                    
                    for k in classes:
                        classes[k].append(preds[k])
        
        return np.array( [ max(classes[i], key=classes[i].count) for i in classes ] )
    
    def score(self, datax, datay):
        """ Calcul du taux de bonne classification en test.
        """
        # Taux de bonne classification sur les données test
        pred_test = self.predict( datax )
        score_test = np.mean( [ 1 if datay[i] == pred_test[i] else 0 for i in range(len( datay )) ] )
        
        return score_test
    
    def getGrid(self):
        return self.grid


class OneVSAll:
    """ Classification multiclasse à partir de classifieurs binaires, méthode
        one vs all.
    """
    def __init__(self, kernel = 'rbf'):
        self.kernel = kernel
    
    def fit(self, xtrain, ytrain):
        """ Phase d'apprentissage. Apprend un classifieur permettant de 
            départager chaque label contre tous les autres labels.
        """
        # Liste de tous les labels du jeu de données
        self.labels = sorted( np.unique( ytrain ) )
        
        # Initialisation de la grille des classifieurs
        self.grid = [ None for i in range( len(self.labels) ) ]
        
        for i in range(len(self.labels)):
            # On isole la classe à départager parmi les autres
            neg = self.labels[i]
                
            # Copie des données
            datax = np.copy(xtrain)
            datay = np.copy(ytrain)
            
            # On remet les labels de la classe neg à -1 et le reste à 1
            datay = np.where(datay==neg, -1, datay)
            datay = np.where(datay!=-1, 1, datay)
                    
            # Mélange des données
            datax, datay = shuffle_data(datax, datay)
            
            # On rajoute à la grille le nouveau classifieur entraîné
            self.grid[i] = SVC(kernel=self.kernel, gamma='auto', probability=True).fit(datax, datay)

    def predict(self, xtest):
        """ Phase de test. Le modèle va chercher la classe qui maximise les
            scores.
        """
        # Dictionnaire des classes prédites pour chaque modèle de self.grid
        classes = { i : [] for i in range(len(xtest)) }
        
        for i in range(len(self.grid)):
            model = self.grid[i]
            scores = model.predict_proba(xtest)
            
            preds = [ self.labels[i] if s[0] > s[1] else -1 for s in scores ]
        
            for k in classes:
                if preds[k] != -1:
                    classes[k].append( preds[k] )
        
        return np.array( [ max(classes[i], key=classes[i].count) if classes[i] != [] else np.random.choice(self.labels) for i in classes ] )
    
    def score(self, datax, datay):
        """ Calcul du taux de bonne classification en test.
        """
        # Taux de bonne classification sur les données test
        pred_test = self.predict( datax )
        score_test = np.mean( [ 1 if datay[i] == pred_test[i] else 0 for i in range(len( datay )) ] )
        
        return score_test
    
    def getGrid(self):
        return self.grid


############################### STRING KERNEL ################################

def find_all_subseqs(string, length_max=2):
    """ Renvoie la liste de toutes les sous-séquences de la chaîne de caractères
        passée en paramètre, de taille <= length_max. Utilise la méthode 
        combinations du module itertools afin de trouver toutes les 
        combinaisons possibles d'indices.
        @param string: str, texte dont on veut trouver toutes les sous-séquences.
    """
    all_inds = [ i for i in range(len(string)) ]
    combs = []
    
    for length in range(1, length_max + 1):
        combs_inds = list(itertools.combinations(all_inds, length))
        combs_strs = list(itertools.combinations(string, length))
        combs += [ ( ''.join(combs_strs[i]) , combs_inds[i][length - 1] - combs_inds[i][0] ) for i in range(len(combs_strs)) ]
        
    return combs

def find_subseqs(string, k=3):
    """ Renvoie la liste de toutes les sous-séquences de la chaîne de caractères
        passée en paramètre, de taille k. Utilise la méthode combinations du 
        module itertools afin de trouver toutes les combinaisons possibles 
        d'indices.
        @param string: str, texte dont on veut trouver toutes les sous-séquences.
    """
    all_inds = [ i for i in range(len(string)) ]

    combs_inds = list(itertools.combinations(all_inds, k))
    combs_strs = list(itertools.combinations(string, k))
    
    return [ ( ''.join(combs_strs[i]) , combs_inds[i][k - 1] - combs_inds[i][0] ) for i in range(len(combs_strs)) ]

def string_kernel(seq1, seq2, lamb=0.7, k=2):
    """ String kernel pour la mesure de similarité entre deux sequences de 
        textes. On ne considère que des séquences de longueur 2.
        @param seq1: str, première séquence
        @param seq2: str, deuxième séquence
        @param lamb: float, entre 0 (on ne tolère pas de gaps) et 1 (des 
                    "occurrences" très éloignées ont le même poids que des 
                    occurences d'une sous-séquence continue)
        @param k: int, longueur des sous-séquences à considérer
    """
    # ----------------- Traitement pour la 1ère séquence --------------------
    
    # Combinaisons d'au plus length_max lettres dans la 1ère séquence
    subseq1 = find_subseqs(seq1, k)
    
    # Combinaisons d'au plus length_max lettres dans la 2e séquence
    subseq2 = find_subseqs(seq2, k)

    # Calcul de similarité entre séquences
    similarite = 0
    
    for s1 in subseq1:
        for s2 in subseq2:
            if s1[0] == s2[0]:
                similarite += lamb ** ( s1[1] + s2[1] )

    return similarite

def matrice_similarite(corpus, lamb=0.7, k=2):
    """ Matrice de similarité sur un corpus de textes.
    """
    M = np.zeros((len(corpus),len(corpus)))
    
    for i in range(len(corpus)):
        for j in range(len(corpus)):
            M[i][j] = string_kernel(corpus[i], corpus[j], lamb, k)
            
    return M

            
##################### EXPERIMENTATIONS ET VISUALISATION ######################

def split_data_mnist(neg, pos):
    """ Chargement des données MNIST et isolation de deux classes neg et pos.
        @param neg: int, première classe à isoler
        @param pos: int, deuxième classe à isoler
    """
    # Chargement des données MNIST
    mnistdatatrain = "data/mnist_train.csv"
    mnistdatatest = "data/mnist_test.csv"
    
    alltrainx,alltrainy = load_mnist(mnistdatatrain)
    alltestx,alltesty = load_mnist(mnistdatatest)
    
    xtrain,ytrain = get_mnist([neg,pos],alltrainx,alltrainy)
    xtest,ytest = get_mnist([neg,pos],alltestx,alltesty)
    
    
    # On remet les labels de la classe neg à -1 et ceux de la classe pos à 1
    ytrain = np.where(ytrain==neg, -1, ytrain)
    ytrain = np.where(ytrain==pos, 1, ytrain)
    ytest = np.where(ytest==neg, -1, ytest)
    ytest = np.where(ytest==pos, 1, ytest)
    
    return xtrain, ytrain, xtest, ytest

def plot_frontiere_proba(data, f, step=20):
    """ Pour visualiser les frontières de décision en 2D.
    """
    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), 255, cmap='Blues_r')
    return plt

def frontiere_decision(model, xtrain, ytrain, xtest, ytest, C=1):
    """ Trace la frontière de décision pour les données et le modèle donnés,
        pour un certain paramètre de régularisation C.
        Le modèle doit être déjà entraîné.
    """
    fig = plt.figure(figsize=(20,5))
    
    # Affichage des données et frontière de décision pour arti2_xtrain
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Score train : %f' % np.mean( cross_val_score(model, xtrain, ytrain, cv=10) ) )
    ax1 = plot_frontiere_proba(xtrain, lambda x: model.predict_proba(x)[:,0], step=20)
    ax1 = plot_data(xtrain, ytrain.reshape(1,-1)[0])
        
    # Affichage des données et frontière de décision pour arti2_xtest
    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Score test : %f' % np.mean( cross_val_score(model, xtest, ytest, cv=10) ) )
    ax2 = plot_frontiere_proba(xtest, lambda x: model.predict_proba(x)[:,0], step=20)
    ax2 = plot_data(xtest, ytest.reshape(1,-1)[0])