from PIL import Image
import os


input_folder = ''
output_folder = ''


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
