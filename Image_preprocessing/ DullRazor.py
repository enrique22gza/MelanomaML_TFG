import os
import cv2

def process_image(input_path, output_path):
    # Leer imagen
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"No se pudo leer la imagen en {input_path}")
        return
    
    # Convertir a escala de grises
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Filtro black hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # Aplicar filtro gaussiano
    gaussian_blur = cv2.GaussianBlur(blackhat, (3, 3), 0)
    
    # Umbral binario (MÁSCARA)
    ret, mask = cv2.threshold(gaussian_blur, 10, 255, cv2.THRESH_BINARY)
    
    # Reemplazar píxeles de la máscara
    clean_image = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)   
    
    # Escribir la imagen procesada en la carpeta de salida
    filename = os.path.basename(input_path)
    output_file = os.path.join(output_path, filename)
    cv2.imwrite(output_file, clean_image)

def process_images_in_folder(input_folder, output_folder):
    # Asegurarse de que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Procesar cada imagen en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Filtrar solo archivos de imagen
            input_path = os.path.join(input_folder, filename)
            process_image(input_path, output_folder)

# Carpetas de entrada y salida
input_folder = ''
output_folder = ''

# Procesar imágenes en la carpeta de entrada y guardar los resultados en la carpeta de salida
process_images_in_folder(input_folder, output_folder)
