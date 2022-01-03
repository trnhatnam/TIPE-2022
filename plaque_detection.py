from math import pi
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter, convolve

chemin = "C:/Users/Utilisateur/OneDrive/projet/"
nom = "test_1.jpg"

img = plt.imread(chemin+nom)
gris = gaussian_filter(0.2*img[:,:,0] + 0.7*img[:,:,1] + 0.1*img[:,:,2], sigma=1) # coef luminescence relative BGR
L, H = gris.shape

### Gradient
print('Calcul du gradient de l\'image...')

sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# On se ramène à une matrice contenant des flottants entre 0 et 1
if np.max(gris) > 1:
    gris = gris/255
else:
    gris = gris/1

"""
# Convolution matricielle
for i in range(L-3):
    for j in range(H-3):
        N = M[i:i+3,j:j+3]
        grad[i][j] = np.sum(np.multiply(sobel_x,N)) + 1j*np.sum(np.multiply(sobel_y,N))
"""

# Convolution matricielle
grad_x = convolve(gris, sobel_x)
grad_y = convolve(gris, sobel_y)
grad = grad_x + 1j*grad_y
G = np.absolute(grad)
L, H = G.shape
theta = np.angle(grad)
seuil = 0.3

print('Seuillage...')

for i in range(L):
    for j in range(H):
        if G[i][j]<seuil:
            G[i][j] = 0.0

print('Déterminer les maxima du gradient...')

Gmax = G.copy()
a = np.arange(pi/8, pi, pi/4)

for j in range(1,L-1):
    for i in range(1,H-1):
        if G[j][i]!=0:
            b = theta[j][i]
            if b>=0:
                if (b<a[0]) or (b>a[3]):
                    g1 = G[j][i-1]
                    g2 = G[j][i+1]
                elif (b<a[1]):
                    g1 = G[j+1][i+1]
                    g2 = G[j-1][i-1]
                elif (b<a[2]):
                    g1 = G[j+1][i]
                    g2 = G[j-1][i]
                else:
                    g1 = G[j+1][i-1]
                    g2 = G[j-1][i+1]
            else:
                if (b<-a[3]):
                    g1 = G[j][i+1]
                    g2 = G[j][i-1]
                elif (b<-a[2]):
                    g1 = G[j-1][i-1]
                    g2 = G[j+1][i+1]
                elif (b<-a[1]):
                    g1 = G[j-1][i]
                    g2 = G[j+1][i]
                elif (b<-a[0]):
                    g1 = G[j-1][i+1]
                    g2 = G[j+1][i-1]
                else:
                    g1 = G[j][i+1]
                    g2 = G[j][i-1]
            if G[j][i]<max(g1,g2):
                Gmax[j][i] = 0.0


for i in range(L):
    for j in range(H):
        if Gmax[i][j] != 0:
            Gmax[i][j] = 1.0

### Contours
print("Recherche du contour...")

contours, _ = cv2.findContours(np.uint8(Gmax), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, reverse=True, key=cv2.contourArea)

for c in contours:
    peri = cv2.arcLength(c, closed=True)
    eps = 0.02 * peri
    approx = cv2.approxPolyDP(c, eps, closed=True)
    if len(approx) == 4:
        # on normalise les plaques en transformant l'éventuelle zone où
        # la plaque est détectée en rectangle
        M = cv2.getPerspectiveTransform(np.float32(approx), np.float32([[0,0], [0,50],[150,50],[150,0]]))
        res = cv2.warpPerspective(img,M,(150,50),flags=cv2.INTER_LINEAR)
        break



cv2.imshow("w",res)
cv2.imwrite("plaque_extraite.jpeg", res)

"""
https://www.f-legrand.fr/scidoc/docmml/image/filtrage/bords/bords.html
https://theailearner.com/tag/cv2-getperspectivetransform/
https://theailearner.com/2019/11/22/simple-shape-detection-using-contour-approximation/
https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
"""