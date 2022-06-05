import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##import

import tensorflow as tf
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as im
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import cv2
from PIL import Image
from tensorflow import keras
from os import chdir as cd
from PIL import Image, ImageFont, ImageDraw
from string import ascii_uppercase
from PIL import ImageOps

## cadre

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.autograph.set_verbosity(0)#éviter pb graph avec tensorflow

assert hasattr(tf, "function")#regarde si utilise bien version 2 tensorflow

kernel = np.ones((3,3),np.uint8)

cd(r'D:\Thomas\TIPE\photo_radar')

lettre_class = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

### base nombre + modèle nombre


train_dir = r'D:\Thomas\TIPE\base_nombre'
validation_dir = r'D:\Thomas\TIPE\base_valid_nombre'

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 16


training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (50, 25),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')


test_set = test_datagen.flow_from_directory(validation_dir,
                                            target_size = (50, 25),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')
## modele

model_chiffre = Sequential()
#convolution
model_chiffre.add(Conv2D(32, (3, 3), padding='same', input_shape = (50, 25, 3),
activation = 'relu'))
model_chiffre.add(Conv2D(32, (3, 3), activation='relu'))
model_chiffre.add(MaxPooling2D(pool_size=(2, 2)))
model_chiffre.add(Dropout(0.5))
# Ajout d'une deuxième couche convolutive
model_chiffre.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model_chiffre.add(Conv2D(64, (3, 3), activation='relu'))
model_chiffre.add(MaxPooling2D(pool_size=(2, 2)))
model_chiffre.add(Dropout(0.5))
# Ajout d'une troisième couche convolutive
model_chiffre.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model_chiffre.add(Conv2D(64, (3, 3), activation='relu'))
model_chiffre.add(MaxPooling2D(pool_size=(2, 2)))
model_chiffre.add(Dropout(0.5)) # antes era 0.25
# Flatten
model_chiffre.add(Flatten())
# Connexion complète
model_chiffre.add(Dense(units = 512, activation = 'relu'))
model_chiffre.add(Dropout(0.5))
model_chiffre.add(Dense(units = 10, activation = 'softmax'))

model_chiffre.summary()


model_chiffre.compile(optimizer = 'rmsprop',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])
## entrainement model + courbes


history = model_chiffre.fit(training_set,
batch_size = 50,
epochs = 10,
validation_data = test_set,
validation_steps = 10)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_chiffres = range(1, len(acc)+1)

plt.plot(epochs_chiffres, acc, 'bo',label = 'training accuracy')
plt.plot(epochs_chiffres, val_acc, 'b', label='Accuracy de validation')
plt.title('Training and validation accuracy')
plt.legend()
plt.plot()
plt.show()

plt.plot(epochs_chiffres, loss, 'bo',label = 'perte accuracy')
plt.plot(epochs_chiffres, val_loss, 'b', label='perte de validation')
plt.title('Training and validation loss')
plt.legend()
plt.plot()
plt.show()

model_chiffre.save('D:\Thomas\TIPE\model_chiffre.h5')

model_chiffre2 =  keras.models.load_model('D:\Thomas\TIPE\model_chiffre.h5')

### base lettres + modèle lettres

train_lettre_dir = r'D:\Thomas\TIPE\base_lettre'
validation_lettre_dir = r'D:\Thomas\TIPE\base_valid_lettre'

train_lettre_datagen = ImageDataGenerator(rescale = 1./255)
test_lettre_datagen = ImageDataGenerator(rescale = 1./255)

batch_size = 16


training_lettre_set = train_lettre_datagen.flow_from_directory(train_lettre_dir,
                                                 target_size = (50, 25),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')


test_lettre_set = test_lettre_datagen.flow_from_directory(validation_lettre_dir,
                                            target_size = (50, 25),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')
##

model_lettre = Sequential()
#convolution
model_lettre.add(Conv2D(32, (3, 3), padding='same', input_shape = (50, 25, 3),
activation = 'relu'))
model_lettre.add(Conv2D(32, (3, 3), activation='relu'))
model_lettre.add(MaxPooling2D(pool_size=(2, 2)))
model_lettre.add(Dropout(0.5))
# Ajout d'une deuxième couche convolutive
model_lettre.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model_lettre.add(Conv2D(64, (3, 3), activation='relu'))
model_lettre.add(MaxPooling2D(pool_size=(2, 2)))
model_lettre.add(Dropout(0.5))
# Ajout d'une troisième couche convolutive
model_lettre.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model_lettre.add(Conv2D(64, (3, 3), activation='relu'))
model_lettre.add(MaxPooling2D(pool_size=(2, 2)))
model_lettre.add(Dropout(0.5)) # antes era 0.25
# Flatten
model_lettre.add(Flatten())
# Connexion complète
model_lettre.add(Dense(units = 512, activation = 'relu'))
model_lettre.add(Dropout(0.5))
model_lettre.add(Dense(units = 26, activation = 'softmax'))

model_lettre.summary()


model_lettre.compile(optimizer = 'rmsprop',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])
##

history_lettre = model_lettre.fit(training_lettre_set,
batch_size = 55,
epochs = 12,
validation_data = test_lettre_set,
validation_steps = 10)


acc_lettre = history_lettre.history['accuracy']
val_acc_lettre = history_lettre.history['val_accuracy']

loss_lettre = history_lettre.history['loss']
val_lettre_loss = history_lettre.history['val_loss']

epochs_lettres = range(1, len(acc_lettre)+1)

plt.plot(epochs_lettres, acc_lettre, 'bo',label = 'training lettre accuracy')
plt.plot(epochs_lettres, val_acc_lettre, 'b', label='Accuracy lettre de validation')
plt.title('Training and validation accuracy')
plt.legend()
plt.plot()
plt.show()

plt.plot(epochs_lettres, loss_lettre, 'bo',label = 'perte lettre accuracy')
plt.plot(epochs_lettres, val_lettre_loss, 'b', label='perte lettre de validation')
plt.title('Training and validation loss')
plt.legend()
plt.plot()
plt.show()

model_lettre.save(r'D:\Thomas\TIPE\model_lettre_final.h5')

model_lettre2 = keras.models.load_model('D:\Thomas\TIPE\model_lettre_final.h5')


### découpage

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
    caractere = []
    caractere.append("borne")
    for i in range(len(M)):
        a =M[i]
        if 0 in a : # regarde si pixel 0 est dans la liste
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
            while ind_temp2 != 'borne':
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

### Prediction


def reco_finale(name,path):
    L = ecriture_finale(name,path)
    lettre_deb =[]
    lettre_fin = []
    nb_mid = []
    for i in range(0,2):
        img_path = L[i]
        img = image.load_img(img_path, target_size=(50, 25))
        blur =  cv2.GaussianBlur(np.float32(img),(3,3),0)
        opening = cv2.morphologyEx(np.float32(blur), cv2.MORPH_OPEN, kernel)
        x = image.img_to_array(opening)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes_lettre = np.argmax(model_lettre2.predict(images, batch_size=10), axis=-1)
        lettre = ''.join(str(elem) for elem in classes_lettre)
        lettre_deb = lettre_deb +[lettre_class[int(lettre)]]
    for j in range (2,5):
        img_path = L[j]
        img = image.load_img(img_path, target_size=(50, 25))
        blur =  cv2.GaussianBlur(np.float32(img),(7,7),0)
        opening = cv2.morphologyEx(np.float32(blur), cv2.MORPH_OPEN, kernel)
        x = image.img_to_array(opening)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = np.argmax(model_chiffre2.predict(images, batch_size=10), axis=-1)
        nb = ''.join(str(elem) for elem in classes)
        nb_mid = nb_mid + [int(nb)]
    for k in range(5,7):
        img_path = L[k]
        img = image.load_img(img_path, target_size=(50, 25))
        blur =  cv2.GaussianBlur(np.float32(img),(3,3),0)
        opening = cv2.morphologyEx(np.float32(blur), cv2.MORPH_OPEN, kernel)
        x = image.img_to_array(opening)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes_lettre = np.argmax(model_lettre2.predict(images, batch_size=10), axis=-1)
        lettre2 = ''.join(str(elem) for elem in classes_lettre)
        lettre_fin = lettre_fin + [lettre_class[int(lettre2)]]
    return(lettre_deb + nb_mid + lettre_fin)
