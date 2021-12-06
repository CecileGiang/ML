# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 08:27:43 2021

@author: GIANG Cécile
"""

#################### IMPORTATIONS LIBRAIRIES ET MODULES ####################

import math
import numpy as np
import pickle

from collections import Counter


################# CHARGEMENT D'UN EXTRAIT DE LA BASE IMDB #################

# data : tableau ( films , features ) , id2titles : dictionnaire id -> titre ,
# fields : id feature -> nom
[ data , id2titles , fields ]= pickle.load( open ("imdb_extrait.pkl","rb"))
# la derniere colonne est le vote
datax = data[: ,:32]
datay = np.array([1 if x [33] >6.5 else -1 for x in data ])

########################### EXERCICE 1: ENTROPIE ###########################

def entropie(vect):
    """ Calcule l'entropie du vecteur vect. L'entropie est d'autant plus 
        grande que l'ensemble est désordonné.
        @param vect: list, objet itérable qui est une liste de labels
        @return : float, entropie correspondant à vect
    """
    # Distribution de probabilités de chaque label dans vect
    p = {key : value/len(vect) for key, value in Counter(vect).items()}
    return sum([-p[i]*math.log2(p[i]) for i in p])


def entropie_cond(liste_vect):
    """ Calcule l'entropie conditionnelle de liste_vect. Dans la pratique, 
        liste_vect est une n-partition de la liste des labels, et la fonction
        calcule donc l'homogénéité de cette partition.
        L'entropie conditionnelle est d'autant plus grande que la partition est
        hétérogène.
        @param liste_vect: list list, partition de labels (liste de listes 
                           de labels)
        @return: float, entropie conditionnelle correspondante
    """
    # p la proportion d'éléments pour chaque partition vect de liste_vect
    length = sum([len(vect) for vect in liste_vect])
    p = [len(vect)/length for vect in liste_vect]
    
    return sum([p[i] * entropie(liste_vect[i]) for i in range(len(liste_vect))])



#################### QUELQUES EXPERIENCES PRELIMINAIRES ####################

from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt

# Construction de l'arbre de décisions
dt = DTree ()
dt.max_depth = 5 # on fixe la taille max de l’arbre à 5
dt.min_samples_split = 2 # nombre minimum d’exemples pour spliter un noeud

# fit: méthode permettant d'apprendre le modèle sur les données passées en paramètre
dt.fit( datax , datay )

# predict: méthode permettant d'obtenir un vecteur de prédictions pour les 
#          données passées en paramètre
dt.predict( datax [:5 ,:])

# score: méthode permettant de renvoyer le taux de bonne classification des
#        données par rapport aux labels passés en paramètre
#print ( dt.score ( datax , datay ))

# Dessin de l ’ arbre dans un fichier pdf si pydot est installé
#dt.to_pdf("/tmp/test_tree.pdf", fields )

# Sinon utiliser http :// www . webgraphviz . com / par exemple ou https :// dreampuf . github . io / GraphvizOnline
#export_graphviz ( dt , out_file ="tree.dot", feature_names = list(fields.values())[:32])


def instanceDT(max_depth=5):
    """ Crée un arbre de décision sur les données X et les labels Y, de
        profoncdeur maximale max_depth. Affiche le taux de bonne classification
        et l'arbre de décision.
        @param max_depth: int, profondeux maximale de l'arbre.
    """
    dt = DTree(max_depth=max_depth, min_samples_split=2)
    dt.fit( datax , datay )
    
    # Affichage du taux de bonne classification
    print ('Taux de bonne classification pour une profondeur de ', max_depth,
           ': ', dt.score(datax, datay))
    
    # Affichage graphique de l'arbre
    plt.figure(figsize=(max_depth * 5, max_depth * 5))
    plot_tree(dt, feature_names= list(fields.values())[:32],filled=True)
    plt.show()



######################### SUR ET SOUS APPRENTISSAGE #########################

def partitionDT(ratio):
    """ Partitionne les données de datax et de datay en deux sous ensembles
        de données étiquetées (train et test) selon le paramètre partage,
        et affiche le graphe des erreurs de classification sur les données test
        et train en fonction de la profondeur de l'arbre.
        @param ratio: list(float, float), indique le ratio de données test-train
    """
    # Mélange de data, création de datax et datay 
    np.random.shuffle(data)
    datax = data[: ,:32]
    datay = np.array([1 if x [33] >6.5 else -1 for x in data ])
    
    # Partition de datax et datay en données d'apprentissage et de de test
    index_split = math.ceil(ratio[0]*len(data)) # indice à partir duquel on sépare les données
    x_train = datax[:index_split]
    x_test = datax[index_split:]
    y_train = datay[:index_split]
    y_test = datay[index_split:]
    
    # Calcul des erreurs de classification pour une profondeur variant de 1 à 25
    erreurs_train = []
    erreurs_test = []
    
    for depth in range(1,26):
        dt = DTree(max_depth=depth, min_samples_split=2)
        dt.fit(x_train, y_train) # entraînement sur les données train
        
        # Calcul des erreurs de classification pour test et train
        erreurs_train.append(1-dt.score(x_train, y_train))
        erreurs_test.append(1-dt.score(x_test, y_test))
    
    # Traçage des courbes d'erreur de classification
    depths = np.arange(1,26)
    
    plt.title('Erreurs de classification en fonction de la profondeur')
    plt.xticks(depths)
    plt.xlabel('Profondeur')
    plt.ylabel('Erreur')
    plt.legend()
    
    plt.plot(depths, erreurs_train, label='Train - {}'.format(ratio[0]), linestyle=':')
    plt.plot(depths, erreurs_test, label='Test - {}'.format(ratio[1]))
    plt.show()



################# VALIDATION CROISEE: SELECTION DE MODELES #################

def validationCroisee(n, display=False):
    """ Réalise une procédure de validation croisée pour des partitions des
        données de data en n sous-ensembles.
        @param n: int, nombre de partitions de data. Les données d'entrainement
                  seront constituées de n-1 sous-ensembles de data, les données
                  de test du reste
        @return erreur: float, erreur moyenne sur les n sous-ensembles pris en 
                        entraînement
    """
    # Mélange de data, création de datax et datay
    np.random.shuffle(data)
    datax = data[: ,:32]
    datay = np.array([1 if x [33] >6.5 else -1 for x in data ])
    
    # Partitionnement de datax et datay en n ensembles, 
    # contenus dans datax_split et datay_split
    datax_split = np.array_split(datax, n)
    datay_split = np.array_split(datay, n)
    
    
    # Calcul des erreurs de classification pour une profondeur variant de 1 à 25
    depth_max = 25
    erreurs_train = [0]*depth_max
    erreurs_test = [0]*depth_max
    
    for j in range(n):
        x_train = np.concatenate(([datax_split[i] for i in range(n) if i != j]))
        x_test = datax_split[j]
        y_train = np.hstack(([datay_split[i] for i in range(n) if i != j]))
        y_test = datay_split[j]
    
        for depth in range(1, depth_max + 1):
            dt = DTree(max_depth=depth, min_samples_split=2)
            dt.fit(x_train, y_train) # entraînement sur les données train
            
            # Calcul des erreurs de classification pour test et train
            erreurs_train[depth-1] += 1-dt.score(x_train, y_train)
            erreurs_test[depth-1] += 1-dt.score(x_test, y_test)
        
        
    # Affichage des erreurs moyennes de classification
        
    if display:
        erreurs_train = [err/n for err in erreurs_train]
        erreurs_test = [err/n for err in erreurs_test]
            
        depths = np.arange(1, depth_max + 1)
            
        plt.title('Erreurs de classification en fonction de la profondeur')
        plt.xticks(depths)
        plt.xlabel('Profondeur')
        plt.ylabel('Erreur')
        plt.legend()
        
        plt.plot(depths, erreurs_train, label='Train', linestyle=':')
        plt.plot(depths, erreurs_test, label='Test')
        plt.show()
    
    # Affichage de l'erreur minimale, ainsi que la profondeur pour laquelle elle
    # est atteinte
    print('Erreur de classification minimale en entraînement: ', min(erreurs_train), 'atteinte pour une profondeur ', erreurs_train.index(min(erreurs_train)))
    print('Erreur de classification minimale en évaluation: ', min(erreurs_test), 'atteinte pour une profondeur ', erreurs_test.index(min(erreurs_test)))