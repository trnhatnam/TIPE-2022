from PIL import Image, ImageFont, ImageDraw
from string import ascii_uppercase
import os
"""
Toutes les polices sont en format .ttf et sont dans le même dossier
L'idée est de chercher toutes les polices (càd fichier.ttf) dans le dossier de travail (donné par os.getcwd()).
Pour chaque police, on met en image chaque lettre et chiffre de cette police qu'on met respectivement dans un dossier pour les lettres et pour les chiffres.
Les fichiers sont des images 25x50 nommés avec cette convention : (charactère)_(numéro).png

Les images seront utilisés pour construire la base des lettres et chiffres pour la reconnaissance des caractères.
On a donc chercher des polices qui ont des similitudes avec les caractères des plaques SIV.
"""



base = ascii_uppercase + '0123456789'
dossierLettre = "images_lettre/"
dossierChiffre = "images_nombre/"
i=0

if not os.path.exists(dossierLettre):
    os.makedirs(dossierLettre)

if not os.path.exists(dossierChiffre):
    os.makedirs(dossierChiffre)

for fichier in os.listdir():
    if os.path.isfile(fichier):
        nom, ext = fichier.split('.')
        if ext in ('ttf','TTF'):
            for char in base:
                police = ImageFont.truetype(fichier, 70)
                L, H = police.getsize(char)
                img = Image.new('RGBA',(L,H), "white")
                dessin = ImageDraw.Draw(img)
                dessin.text((0, 0), char,(0, 0, 0),font=police)
                img = img.resize((25,50))
                if char.isdigit():
                    img.save(dossierChiffre + char + '%s.png' % (i))
                else:
                    img.save(dossierLettre + char + '%s.png' % (i))
            i += 1

