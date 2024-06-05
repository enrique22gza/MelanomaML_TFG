#sobremuestreo NO_melanoma - CREA 5 IMAGENES POR CADA IMAGEN ORIGINAL=680
import cv2
import numpy as np
import os

# Directorio de entrada y salida
input_folder = ''
output_folder = ''
os.makedirs(output_folder, exist_ok=True)

# Función para realizar rotación, zooming, shifting & cropping
def augment_image(image):
    augmented_images = []

    for angle in [0, 45, 270]:  
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        
        # Calcular la matriz de rotación
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        
        # Realizar la rotación
        rotated_image = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated_image)



    # Shifting
    for dx, dy in [(50, 0),(0,-50)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(shifted_image)

    

    return augmented_images

# Iterar sobre las imágenes en el directorio de entrada
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_prefix = os.path.splitext(filename)[0]

        # Leer la imagen
        image = cv2.imread(input_path)

        # Aplicar aumentos
        augmented_images = augment_image(image)

        # Guardar las imágenes aumentadas
        for i, augmented_image in enumerate(augmented_images):
            output_filename = f'{output_prefix}_aug_{i}.png'
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, augmented_image)
