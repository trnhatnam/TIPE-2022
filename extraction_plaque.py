import numpy as np
from skimage import filters
from scipy.signal import convolve2d
import cv2

chemin = "C:/Users/Utilisateur/OneDrive/projet/"
source = "test_6.jpeg"

### Détection des contours

# Conversion en niveau de gris
img = cv2.imread(chemin+source)
gris = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3,3), 1)
L,H = gris.shape

# Convolution matricielle avec les filtres de Sobel
fy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
fx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
gx = convolve2d(gris, fx, mode="same", boundary="fill", fillvalue=0)
gy = convolve2d(gris, fy, mode="same", boundary="fill", fillvalue=0)
z = gx+1j*gy
grad = np.absolute(z)
angle = np.angle(z)

# Raffinement des contours
Gmax = np.zeros((L,H))
a = np.arange(np.pi/8, np.pi, np.pi/4) # array des "angles moitiés"

for i in range(1,L-1):
    for j in range(1,H-1):
        if grad[i][j]!=0:
            b = angle[i][j]
            if b>=0:
                if (b<a[0]) or (b>a[3]):
                    g1 = grad[i][j-1]
                    g2 = grad[i][j+1]
                elif (b<a[1]):
                    g1 = grad[i+1][j+1]
                    g2 = grad[i-1][j-1]
                elif (b<a[2]):
                    g1 = grad[i+1][j]
                    g2 = grad[i-1][j]
                else:
                    g1 = grad[i+1][j-1]
                    g2 = grad[i-1][j+1]
            else:
                if (b<-a[3]):
                    g1 = grad[i][j+1]
                    g2 = grad[i][j-1]
                elif (b<-a[2]):
                    g1 = grad[i-1][j-1]
                    g2 = grad[i+1][j+1]
                elif (b<-a[1]):
                    g1 = grad[i-1][j]
                    g2 = grad[i+1][j]
                elif (b<-a[0]):
                    g1 = grad[i-1][j+1]
                    g2 = grad[i+1][j-1]
                else:
                    g1 = grad[i][j+1]
                    g2 = grad[i][j-1]
            if grad[i][j]>=max(g1,g2):
                Gmax[i][j] = grad[i][j]

# Seuillage par hystérésis
med = np.median(gris)
bords = np.uint8(filters.apply_hysteresis_threshold(Gmax, 0, med))


### 1er niveau : Recherche de contours
cts, hierarchy = cv2.findContours(bords, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# CC_COMP : on hiérarchise les contours dans deux catégories : les contours ouverts et fermés
# APPROX_NONE : on ne fait pas d'approximation des contours càd au lieu qu'une ligne soit décrit par deux points, elle sera décrite par tous les points qui la composent
fermes = [c for c, h in zip(cts, hierarchy[0]) if h[2] != -1]
fermes.sort(key=cv2.contourArea, reverse=True)

### 2è niveau : transformation de Hough

blanc = np.zeros((L,H)) # tableau blanc
candidats = []

for ct in fermes:
    blanc[blanc != 0] = 0 # on efface le tableau
    cv2.drawContours(blanc, ct, -1, (255,255,255),1)
    lines = cv2.HoughLinesWithAccumulator(np.uint8(blanc), 1, np.pi/180, 0)

    max_ind = sorted(lines, key=lambda x : x[0][2], reverse=True)[:2] # on prend les deux droites qui passent par le plus de points constituant le contour
    rho1,theta1, _ = max_ind[0][0]
    rho2,theta2, _ = max_ind[1][0]
    epsTheta = 3*np.pi/180
    epsRho = 10
    if abs(theta1-theta2) <= epsTheta and abs(rho1-rho2) >= epsRho: # si ses contours sont parallèles et sont bien espacées
        candidats.append(ct)

trouvé = False

for ct in candidats:
    _,_,w,h = cv2.boundingRect(ct)
    if w < h:
        continue # cas rectangle "verticale" = pas intéréssant

    # on prend les points en haut à droit, haut à gauche et en bas à gauche
    rangement1 = sorted(ct, key=lambda x : x[0][0] + x[0][1])
    topleft, botright = rangement1[0], rangement1[-1]
    botleft = sorted(ct, key=lambda x : x[0][1] - x[0][0])[-1]
    M = cv2.getAffineTransform(np.float32([topleft[0], botleft[0], botright[0]]), np.float32([[0,0], [0,50], [150,50]]))
    echantillon = cv2.warpAffine(gris, M, (150,50))

    # seuillage noir/blanc
    med = np.median(echantillon)
    echantillon[echantillon <= 0.33*med] = 0.0
    echantillon[echantillon > 0.33*med] = 255.0
    bande = echantillon[25]

    # comptage
    compte = 0
    initial = bande[0]
    i = 0
    for val in bande:
        if val != initial:
            if initial == 0:
                compte += 1
            initial = val
    inf = 7
    sup = 24
    if inf <= compte <= sup:
        print('trouvé')
        trouvé = True
        cv2.drawContours(img, ct, -1, (0,0,255), 2) # seulement pour vérifier si on a pris le bon candidat
        cv2.imshow('w', echantillon)
        cv2.imwrite("plaque_"+source,echantillon)
        break


if not trouvé:
    print('pas trouvé')








