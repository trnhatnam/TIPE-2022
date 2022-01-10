import tensorflow
from os import chdir as cd
import pytesseract
from PIL import Image
import PIL.ImageOps
import os
import cv2
import argparse
from os import chdir as cd
import numpy as np
import io
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
from PIL import Image
from PIL import ImageOps
import keras
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

"""Mettre 'C:\\adresse de tesseract.exe' """
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract- OCR\\tesseract.exe'


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cd('D:\Thomas\TIPE\photo_radar')

### PRETRAITEMENT
## modèles lettres + chiffres + bases

"base chiffres + modèle"

def base_train(path):
    n=9
    X_basej= []
    Y_basej = []
    for i in range (0,n+1) :
        for j in range(1,9):
            img = Image.open(path+"\\"+str(i)+"."+str(j)+".png")
            img.load()
            img= np.asarray(img, dtype=np.uint8)
            X_basej.append(img)
            Y_basej.append(i)
    return X_basej,Y_basej

X_p,Y_p = base_train(r'D:\Thomas\TIPE\base_nombre')


X_train = X_p
Y_train = Y_p

def base_test(path):
    n=9
    X_basej= []
    Y_basej = []
    for i in range (0,n+1) :
        for j in range(1,3):
            img = Image.open(path+"\\"+str(i)+"."+str(j)+".png")
            img.load()
            img= np.asarray(img, dtype=np.uint8)
            X_basej.append(img)
            Y_basej.append(i)
    return X_basej,Y_basej

X_test,Y_test = base_test(r'D:\Thomas\TIPE\base_test_nombre')
class creation_model():
    def __init__(self):
        self.Model= tf.keras.Sequential()

        # 3 couches de convolution, avec Nb filtres progressif 32, 64 puis 128
        self.Model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(55, 55, 4), activation='relu'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))

        self.Model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(55, 55, 4), activation='relu'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))

        self.Model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(55, 55, 4), activation='relu'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))

        # remise à plat
        self.Model.add(Flatten())

        # Couche dense classique ANN
        self.Model.add(Dense(512, activation='relu'))

        # Couche de sortie (classes de 0 à 9)
        self.Model.add(Dense(10, activation='softmax'))

modelf_chiffre = creation_model()

modelf_chiffre.Model.compile(
loss = "sparse_categorical_crossentropy",
optimizer="adam",
metrics = ["accuracy"]
)

modelf_chiffre.Model.summary()


early_stop_chiffre = EarlyStopping(monitor='val_accuracy', mode = 'max', patience = 10)

modelf_chiffre.Model.fit(x=np.array(X_train),
                 y =np.array(Y_train),
                 validation_data = (np.array(X_test),np.array(Y_test)),
                 epochs=80,
                 callbacks=[early_stop_chiffre])

losses = pd.DataFrame(modelf_chiffre.Model.history.history)
losses[['accuracy', 'val_accuracy']].plot()
plt.show()

"base lettres + modèle"

def base_trainl(path):
    n=26
    X_basej= []
    Y_basej = []
    lettres = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    for i in range (0,n) :
        for j in range(1,5):
            img = Image.open(path+"\\"+lettres[i]+str(j)+".png")
            img.load()
            img= np.asarray(img, dtype=np.uint8)
            X_basej.append(img)
            Y_basej.append(i)
    return X_basej,Y_basej

X_p,Y_p = base_trainl(r'D:\Thomas\TIPE\base_lettre')


X_train_lettre = X_p
Y_train_lettre = Y_p

def base_testl(path):
    n=26
    X_basej= []
    Y_basej = []
    lettres = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    for i in range (0,n) :
        for j in range(1,3):
            img = Image.open(path+"\\"+lettres[i]+str(j)+".png")
            img.load()
            img= np.asarray(img, dtype=np.uint8)
            X_basej.append(img)
            Y_basej.append(i)
    return X_basej,Y_basej

X_test_lettre,Y_test_lettre = base_testl(r'D:\Thomas\TIPE\base_test_lettre')

class creation_model3():
    def __init__(self):
        self.Model= tf.keras.Sequential()

        # 3 couches de convolution, avec Nb filtres progressif 32, 64 puis 128
        self.Model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(55, 55, 4), activation='relu'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))

        self.Model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(55, 55, 4), activation='relu'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))

        self.Model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(55, 55, 4), activation='relu'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))

        # remise à plat
        self.Model.add(Flatten())

        # Couche dense classique ANN
        self.Model.add(Dense(512, activation='relu'))

        # Couche de sortie (classes de 0 à 9)
        self.Model.add(Dense(27, activation='softmax'))



modelf_lettre = creation_model3()

modelf_lettre.Model.compile(
loss = "sparse_categorical_crossentropy",
optimizer="adam",
metrics = ["accuracy"]
)

modelf_lettre.Model.summary()


early_stop = EarlyStopping(monitor='accuracy', mode = 'max', patience = 10)

modelf_lettre.Model.fit(x=np.array(X_train_lettre),
                 y =np.array(Y_train_lettre),
                 validation_data = (np.array(X_test_lettre),np.array(Y_test_lettre)),
                 epochs=208,
                callbacks=[early_stop])


losses = pd.DataFrame(modelf_lettre.Model.history.history)
losses[['accuracy', 'val_accuracy']].plot()
plt.show()

'base forme + modèle'

def base_trainf(path):
    X_basej= []
    Y_basej = []
    for j in range(1,9):
        img = Image.open(path+"\\"+'cercle'+str(j)+".png")
        img.load()
        img= np.asarray(img, dtype=np.uint8)
        X_basej.append(img)
        Y_basej.append(1)
    for k in range(1,9):
        img = Image.open(path+"\\"+'rectangle'+str(j)+".png")
        img.load()
        img= np.asarray(img, dtype=np.uint8)
        X_basej.append(img)
        Y_basej.append(0)
    return X_basej,Y_basej

X_p,Y_p = base_trainf(r'D:\Thomas\TIPE\base_forme')


X_train_forme = X_p
Y_train_forme = Y_p

def base_testf(path):
    X_basej= []
    Y_basej = []
    for j in range(1,5):
        img = Image.open(path+"\\"+'cercle'+str(j)+".png")
        img.load()
        img= np.asarray(img, dtype=np.uint8)
        X_basej.append(img)
        Y_basej.append(1)
    for k in range(1,5):
        img = Image.open(path+"\\"+'rectangle'+str(j)+".png")
        img.load()
        img= np.asarray(img, dtype=np.uint8)
        X_basej.append(img)
        Y_basej.append(0)
    return X_basej,Y_basej

X_test_forme,Y_test_forme = base_testf(r'D:\Thomas\TIPE\base_test_forme')

class creation_model4():
    def __init__(self):
        self.Model= tf.keras.Sequential()

        # 3 couches de convolution, avec Nb filtres progressif 32, 64 puis 128
        self.Model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(150, 150, 4), activation='relu'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))

        self.Model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150, 150, 4), activation='relu'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))

        self.Model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150, 150, 4), activation='relu'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))

        # remise à plat
        self.Model.add(Flatten())

        # Couche dense classique ANN
        self.Model.add(Dense(512, activation='relu'))

        # Couche de sortie (classes de 0 à 9)
        self.Model.add(Dense(2, activation='softmax'))

modelf_forme = creation_model4()

modelf_forme.Model.compile(
loss = "sparse_categorical_crossentropy",
optimizer="adam",
metrics = ["accuracy"]
)

modelf_forme.Model.summary()

early_stop_forme = EarlyStopping(monitor='accuracy', mode = 'max', patience = 5)

modelf_forme.Model.fit(x=np.array(X_train_forme),
                 y =np.array(Y_train_forme),
                 validation_data = (np.array(X_test_forme),np.array(Y_test_forme)),
                 epochs=16,
                callbacks=[early_stop_forme])


losses = pd.DataFrame(modelf_forme.Model.history.history)
losses[['accuracy', 'val_accuracy']].plot()
plt.show()

## extraction d'image

#chez nhat nam

### DECOUPAGE
#découpage

## découpage verical


def decoup_verti_ligne_mat(path):
    image =Image.open(path,'r')
    img = image.convert('L')
    (L,H)=img.size
    pix =img.load()
    pix_t = [H*[0] for i in range (L)]
    for i in range(L):
        for j in range(H):
            pix_t[i][j] = img.getpixel((i,j))#récupère le pixel
    return pix_t # récupère matrice



def trie_caracteres(path):
    M = decoup_verti_ligne_mat(path)
    a=[]
    b=0
    liste = []
    caractere = []
    caractere.append("borne")
    for i in range(len(M)):
        a =M[i]
        if 255 in a : # regarde si pixel 0 est dans la liste
            caractere.append(a)
        else :
            if caractere[i-b-1]== 'borne':
                b+=1
            else :
                caractere.append("borne") # délimite la postion d'un caractère en insérant le mot borne où l'on a blanc
    caractere.append("borne")
    return caractere

def ecriture_fichiers(name,path):# écrire chaque ligne dans un fichier séparé
    M = trie_caracteres(path)
    mat = decoup_verti_ligne_mat(path)
    n = len(M)
    ind_temp = 0
    ind_temp2 = 0
    j = 0
    indicatrice = []
    for i in range(n-1):
        ind_temp = M[i]
        d_liste = []
        if ind_temp == 'borne':
            j = i +1
            ind_temp2 = M[j]
            while ind_temp2 != 'borne': # regarde après avoir trouvé la 1ere borne
                d_liste.append(ind_temp2)
                j+=1
                ind_temp2 = M[j]
            if d_liste != [] :
                ligne = np.array(d_liste, dtype=np.uint8)# uint 8 : u -> unsigned int -> traite comme un nombre 8 -> nb de bits
                nv_fichier = Image.fromarray(ligne) # transforme la matrice de pixels en image
                nv_fichier.save(name+str(i)+".png")
                indicatrice.append("D:\Thomas\TIPE\photo_radar" + "\\" + name + str(i) + '.png')
            ind_temp = ind_temp2
    return indicatrice


def ecriture_finale(name,path):
    L = ecriture_fichiers(name,path)
    indicatrice = []
    for k in range(len(L)):
        image_origine = Image.open(L[k])
        image_tourne =image_origine.transpose(Image.ROTATE_270) # retourne l'image de 270 degrés
        image_finale = ImageOps.mirror(image_tourne)
        image_finale.save("D:\Thomas\TIPE\photo_radar"+"\\"+name+"f"+str(k)+'.png')
        indicatrice.append("D:\Thomas\TIPE\photo_radar"+"\\"+name+"f"+str(k)+'.png')
    return indicatrice

### identification plaque

    lettres = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

img =Image.open("D:\Thomas\TIPE\photo_radar\caractèresf6.png")
img2 = np.array(img)
img_f = cv2.resize(img2,(110,110))
img_f2 = img_f.reshape(1,55,55,4)


modelf_lettre.Model.predict(img_f2)
lettres[np.argmax(modelf_lettre.Model.predict(img_f2), axis=-1)[0]]

def reco_lettres(name,path):
    lettres = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    L = ecriture_finale(name,path)
    lettres_plaque_deb = []
    lettres_plaque_fin =[]
    for i in range(0,2):
        p = 0
        m = 0
        img = L[i]
        img = Image.open(L[i])
        img2 = np.array(img)
        img_temp = cv2.resize(img2,(110,110))
        img_f = img_temp.reshape(1,55,55,4)
        m = modelf_lettre.Model.predict(img_f)
        p = lettres[np.argmax(m, axis = -1)[0]]
        lettres_plaque_deb.append(p)
    for j in range (7,9):
        p = 0
        m = 0
        img = L[i]
        img = Image.open(L[i])
        img2 = np.array(img)
        img_temp = cv2.resize(img2,(110,110))
        img_f = img_temp.reshape(1,55,55,4)
        m = modelf_lettre.Model.predict(img_f)
        p = lettres[np.argmax(m, axis = -1)[0]]
        lettres_plaque_fin.append(p)
    return lettres_plaque_deb,lettres_plaque_fin



def reco_chiffres(name,path):
    L = ecriture_finale(name,path)
    chiffres_plaque = []
    for i in range (3,6):
        chiffre = 0
        img = Image.open(L[i])
        img2 = np.array(img)
        img_temp = cv2.resize(img2,(110,110))
        img_f = img_temp.reshape(1,55,55,4)
        m = modelf_chiffre.Model.predict(img_f)
        chiffre = np.argmax(m, axis=-1)[0]
        chiffres_plaque.append(chiffre)
    return chiffres_plaque

def reco_forme(name,path):
    L = ecriture_finale(name,path)
    forme_plaque = []
    forme = ["-","o"]
    f1 = L[2]
    img = Image.open(f1)
    img2 = np.array(img)
    img_temp = cv2.resize(img2,(300,300))
    img_f = img_temp.reshape(1,150,150,4)
    m = modelf_forme.Model.predict(img_f)
    p = forme[np.argmax(m, axis = -1)[0]]
    forme_plaque.append(p)
    f2 = L[6]
    img_f = Image.open(f2)
    img_f2 = np.array(img_f)
    img_temp_f = cv2.resize(img_f2,(300,300))
    img_ff = img_temp_f.reshape(1,150,150,4)
    m = modelf_forme.Model.predict(img_ff)
    n = forme[np.argmax(m, axis = -1)[0]]
    forme_plaque.append(n)
    return forme_plaque

def reco_finale(name,path):
    K1,K2 = reco_lettres(name,path)
    L = reco_chiffres(name,path)
    M = reco_forme(name,path)
    plaque_lu =''
    for i  in range (2):
        plaque_lu += K1[i]
    plaque_lu += '-'
    for j in range(3):
        plaque_lu +=  str(L[j])
    plaque_lu += '-'
    for k in range(2):
        plaque_lu += K2[k]
    return plaque_lu







