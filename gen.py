from PIL import Image, ImageFont, ImageDraw
from string import ascii_uppercase
import os

# toutes les polices sont dans le même dossier

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

