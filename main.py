import numpy as np
import activation_functions
from PIL import Image, ImageChops

rng = np.random.default_rng()

#On definit les reseaux de neuronnes

class MLP:

    def __init__(self, couches_tailles):  # couches_tailles est la liste des longueurs de chaque couche
        self.taille = couches_tailles
        self.nb_couche = len(couches_tailles)  # nb de couches
        self.poids = [rng.normal(0, 1, (y, x)) for x, y in zip(couches_tailles[:-1], couches_tailles[1:])]  # tableau de tableaux comportant les poids des arêtes entre la couche i et i + 1
        self.biais = [rng.normal(0, 1, (x, 1)) for x in couches_tailles[1:]]  # tableau comportant tous les biais

    def propagation_directe(self, a):
        a = a.reshape(-1, 1)
        for p, b in zip(self.poids, self.biais):
            a = activation_functions.sigmoide(np.dot(p, a) + b)
        return a

    def deriv_cout(self, donnees_de_sortie, y):
        return donnees_de_sortie - y

    def propagation_contraire(self, x, y):
        nabla_p = [np.zeros(np.shape(p)) for p in self.poids]
        nabla_b = [np.zeros(np.shape(b)) for b in self.biais]
        a = x.reshape(-1, 1)
        activations = [x]
        zlist = []
        for p, b in zip(self.poids, self.biais):
            z = np.dot(p, a) + b
            zlist.append(z)
            a = activation_functions.sigmoide(z)
            activations.append(a)

        delta = self.deriv_cout(activations[-1], y) * activation_functions.deriv_sigmoide(zlist[-1])
        nabla_p[-1] = np.dot(delta, activations[-2].T)
        nabla_b[-1] = delta

        for i in range(2, self.nb_couche):
            z = zlist[-i]
            delta = np.dot(self.poids[-i + 1].T, delta) * activation_functions.deriv_sigmoide(z)
            nabla_p[-i] = np.dot(delta, activations[-i - 1].T)
            nabla_b[-i] = delta

        return (nabla_p, nabla_b)

    def actualisation_echantillon(self, echantillon, eta):
        nabla_p = [np.zeros(np.shape(p)) for p in self.poids]
        nabla_b = [np.zeros(np.shape(b)) for b in self.biais]
        for x, y in echantillon:
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            dnp, dnb = self.propagation_contraire(x, y)
            nabla_p = [a + b for a, b in zip(nabla_p, dnp)]
            nabla_b = [a + b for a, b in zip(nabla_b, dnb)]
        self.poids = [p - (eta/len(echantillon)) * nablap for p, nablap in zip(self.poids, nabla_p)]
        self.biais = [b - (eta / len(echantillon)) * nablab for b, nablab in zip(self.biais, nabla_b)]

    def SGD(self, donnees_entrainement, epochs, longueur_echantillon, eta, test_donnees = None):
        if test_donnees:
            test_donnees = list(test_donnees)
            ntest_donnees = len(test_donnees)
        n = len(donnees_entrainement)
        for i in range(epochs):
            rng.shuffle(donnees_entrainement)
            echantillons = [donnees_entrainement[j : j + longueur_echantillon] for j in range(0, n, longueur_echantillon)]
            for ech in echantillons:
                self.actualisation_echantillon(ech, eta)
            if test_donnees:
                print("Epoch {}: {} / {}".format(i, self.evaluer(test_donnees), ntest_donnees))
            print("Epoch {} sur {}".format(i, epochs))

    def evaluer(self, donnees):
        res = [(np.argmax(self.propagation_directe(x)), y) for x, y in donnees]
        s = sum(int(x==y) for x, y in res)
        return s

#Optimisation de l'entrainement
'''
def applyImageEffect(arr, rot, shiftx, shifty):
    return np.asarray(
        ImageChops.offset(
            Image.fromarray(arr.reshape((28, 28)) * 255.)
                .rotate(rot)
        , shiftx, shifty)
    ).reshape(28 ** 2) / 255.

def applyRandomEffect(arr, rotation_ampl = 45, shift_ampl = 5):
    return applyImageEffect(
        arr,
        np.random.uniform(-rotation_ampl, rotation_ampl),
        np.random.randint(-shift_ampl, shift_ampl),
        np.random.randint(-shift_ampl, shift_ampl))

def generateRandomData(img, labels, n):
    assert len(img) == len(labels)

    gen_img = np.empty((n, *img[0].shape))
    gen_labels = np.empty(n, dtype = np.uint32)

    for i in range(n):
        gen_img[i] = applyRandomEffect(img[i % len(img)], 30, 5)
        gen_labels[i] = labels[i % len(img)]

    return gen_img, gen_labels

#Chargement des donnees

def loadTrainingData(n, imagefile, labelfile, startFrom = 0):
    IMAGE_SIZE = 28 ** 2

    with open(imagefile, "rb") as f:
        f.read(16)# Header : Magic + i_img + i_row + i_col
        if startFrom:
            f.read(IMAGE_SIZE * startFrom)
        images = np.frombuffer(f.read(IMAGE_SIZE * n), dtype = np.uint8) \
            .astype(np.float32) \
            .reshape(n, IMAGE_SIZE) / 255.0

    with open(labelfile, "rb") as f:
        f.read(8)# Header : Magic + i_items
        if startFrom:
            f.read(startFrom)

        labels = np.frombuffer(f.read(n), dtype = np.uint8) \
            .reshape(n)

    return images, labels
'''

