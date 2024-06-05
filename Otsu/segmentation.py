import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Rutas de las imágenes y resultados
ruta_imagenes = ''
ruta_resultados = ''

os.makedirs(ruta_resultados, exist_ok=True)

# Función para aplicar el umbral de Otsu e invertir la máscara
def aplicar_otsu_invertido(imagen):
    # Convierte la imagen a escala de grises si es necesario
    if len(imagen.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen
    
    # Aplica el umbral de Otsu
    _, imagen_otsu = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invierte la máscara (región de interés en blanco, fondo en negro)
    imagen_otsu_invertida = cv2.bitwise_not(imagen_otsu)
    
    return imagen_otsu_invertida

# Obtiene la lista de archivos de imágenes
archivos_imagenes = [f for f in os.listdir(ruta_imagenes) if f.endswith('.png')]

# Selecciona 5 imágenes al azar para mostrar
muestras_aleatorias = random.sample(archivos_imagenes, 5)

# Itera sobre cada imagen y aplica el umbral de Otsu invertido
for archivo_imagen in archivos_imagenes:
    # Carga la imagen
    imagen = cv2.imread(os.path.join(ruta_imagenes, archivo_imagen))
    
    if imagen is None:
        print(f"Imagen no encontrada: {archivo_imagen}")
        continue
    
    # Aplica el umbral de Otsu a la imagen e invierte la máscara
    imagen_otsu_invertida = aplicar_otsu_invertido(imagen)
    
    # Guarda la imagen segmentada como máscara
    nombre_mascara = archivo_imagen.replace('.png', '_mascara_otsu_invertida.png')
    ruta_resultado = os.path.join(ruta_resultados, nombre_mascara)
    cv2.imwrite(ruta_resultado, imagen_otsu_invertida)
    
    # Muestra solo 5 imágenes al azar
    if archivo_imagen in muestras_aleatorias:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title('Imagen Original')
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('Imagen en Gris')
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY), cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('Máscara Segmentada (Otsu Invertido)')
        plt.imshow(imagen_otsu_invertida, cmap='gray')
        plt.axis('off')
        
        plt.show()

print("Segmentación completada.")
