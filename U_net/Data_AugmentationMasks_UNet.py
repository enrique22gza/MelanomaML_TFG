#sobremuestreo U-NET MASCARAS-> CREA 5 IMAGENES POR CADA IMAGEN ORIGINAL=850

import os
import numpy as np
import cv2

def rotate_image(image, angle):
    """Rota una imagen en un ángulo dado."""
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # Rotación
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated

def shift_image(image, x, y):
    """Desplaza una imagen en x e y."""
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return shifted

def augment_and_save_masks(input_folder, output_folder, angles, shifts):
    """Aplica rotación y desplazamiento a máscaras en una carpeta y guarda los resultados en otra."""
    # Crear el directorio de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Recorrer todos los archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".bmp"):
            file_path = os.path.join(input_folder, filename)
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            output_prefix = os.path.splitext(filename)[0]

            i = 0
            # Rotaciones
            for angle in angles:
                rotated = rotate_image(mask, angle)
                output_filename = f"{output_prefix}_aug_{i}.png"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, rotated, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                i += 1
            
            # Desplazamientos
            for (x_shift, y_shift) in shifts:
                shifted = shift_image(mask, x_shift, y_shift)
                output_filename = f"{output_prefix}_aug_{i}.png"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, shifted, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                i += 1


input_folder = ''
output_folder = ''
angles = [0, 90, 270]
shifts = [(50, 0), (-50, 0)]

augment_and_save_masks(input_folder, output_folder, angles, shifts)
