import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

chemin = "C:/Users/Utilisateur/OneDrive/projet"
nom = "plaque_extraite_test_6.jpeg"
os.chdir(chemin)
source = plt.imread(nom)
if np.max(source) > 1:
    source = source*1
else:
    source = source*255.0
L, H, _ = source.shape
img = np.zeros((L,H))

for i in range(L):
    for j in range(H):
        V = source[i][j]
        if 0.2*V[0]+0.7*V[1]+V[2]*0.1  <= 70:
            img[i][j] = 255.0
        else:
            img[i][j] = 0.0

cv2.imshow("w",img)
cv2.imwrite("resultat_2.jpeg", img)
