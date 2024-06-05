#DIVIDIR TRAIN EN IMAGENES Y MASCARAS

import os
import shutil
import re

directorio_raiz = ''

# Directorios de salida para las imágenes y las máscaras
directorio_imagenes = ''
directorio_mascaras = ''

if not os.path.exists(directorio_imagenes):
    os.makedirs(directorio_imagenes)

if not os.path.exists(directorio_mascaras):
    os.makedirs(directorio_mascaras)


patron = re.compile(r'IMD\d{3}')

# Recorre la estructura de carpetas
for carpeta_raiz, carpetas, archivos in os.walk(directorio_raiz):
    for carpeta in carpetas:
        if patron.match(carpeta):
            # Directorio de la lesión
            directorio_lesion = os.path.join(carpeta_raiz, carpeta)
            
            # Copiar imágenes al dataset de imágenes
            for subcarpeta in os.listdir(directorio_lesion):
                if subcarpeta.endswith('_Dermoscopic_Image'):
                    directorio_imagen = os.path.join(directorio_lesion, subcarpeta)
                    for imagen in os.listdir(directorio_imagen):
                        if os.path.isfile(os.path.join(directorio_imagen, imagen)):
                            shutil.copy(os.path.join(directorio_imagen, imagen), directorio_imagenes)
                            
            
            # Copiar máscaras al dataset de máscaras
            for subcarpeta in os.listdir(directorio_lesion):
                if subcarpeta.endswith('_lesion'):
                    directorio_mascara = os.path.join(directorio_lesion, subcarpeta)
                    for mascara in os.listdir(directorio_mascara):
                        if os.path.isfile(os.path.join(directorio_mascara, mascara)):
                            shutil.copy(os.path.join(directorio_mascara, mascara), directorio_mascaras)
