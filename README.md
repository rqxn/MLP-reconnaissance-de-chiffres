# Motivation
Ce projet a été réalisé en suivant l'excellente [playlist de 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) sur les réseaux de neurones, elle-même basée sur les travaux de [Michael Nielsen](https://michaelnielsen.org/).

# Description
Ici, on n'étudie pour une introduction qu'un MLP, dont on peut faire varier le nombre de couches, la fonction d'activation utilisée, etc. Ce MLP a pour première couche 784 neurones, correspondant aux 784 pixels d'une image de taille 28*28 qu'il reçoit en entrée. Sa dernière couche a 10 neurones, représentant les 10 chiffres que l'on souhaite identifier.

Le réseau de neurones est entrainé à l'aide de l'algorithme du gradient stochastique ainsi que d'échantillons de tailles arbitraires issus de l'[ensemble de données MNIST](https://yann.lecun.com/exdb/mnist/).

# Remarques
Pour des soucis de simplicité et n'étant pas le coeur du projet, le fichier permettant de charger les données de MNIST a été tiré du travail de Michael Nielsen. 

Les résultats et le taux de succès sont ceux attendus avec des paramètres de référence (c'est-à-dire un réseau de neurones ayant trois couches de tailles 784, 30 et 10, 30 epochs, une longueur d'échantillon de 10 et un taux d'apprentissage de 3).

Ce travail n'a pour objectif que d'être une introduction au sujet, et ne cherche donc pas à être optimal. 
