from PIL import Image
import os

# Define la carpeta donde están tus imágenes originales
input_folder = ''

# Define la carpeta donde quieres guardar las imágenes redimensionadas
output_folder = ''

# Si la carpeta de salida no existe, la crea
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Itera sobre todos los archivos en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')): 
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        
        # Cambia el tamaño de la imagen
        img_resized = img.resize((224, 224))
        
        # Guarda la imagen redimensionada en la carpeta de salida
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)
