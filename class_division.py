import os
import shutil

#carpetas donde estan los datos de entrenamiento, validación y prueba
folders = ['']

melanoma_images = set([
    'IMD058.png', 'IMD061.png', 'IMD063.png', 'IMD064.png', 'IMD065.png',
    'IMD080.png', 'IMD085.png', 'IMD088.png', 'IMD090.png', 'IMD091.png',
    'IMD168.png', 'IMD211.png', 'IMD219.png', 'IMD240.png', 'IMD242.png',
    'IMD284.png', 'IMD285.png', 'IMD348.png', 'IMD349.png', 'IMD403.png',
    'IMD404.png', 'IMD405.png', 'IMD407.png', 'IMD408.png', 'IMD409.png',
    'IMD410.png', 'IMD413.png', 'IMD417.png', 'IMD418.png', 'IMD419.png',
    'IMD406.png', 'IMD411.png', 'IMD420.png', 'IMD421.png', 'IMD423.png',
    'IMD424.png', 'IMD425.png', 'IMD426.png', 'IMD429.png', 'IMD435.png'
])

# Crear las subcarpetas si no existen
for folder in folders:
    os.makedirs(os.path.join(folder, 'melanoma'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'no_melanoma'), exist_ok=True)

# Mover las imágenes a las subcarpetas correspondientes
for folder in folders:
    for image in os.listdir(folder):
        if image.endswith('.png'):  # Asegúrate de que solo mueves imágenes
            source_path = os.path.join(folder, image)
            if image in melanoma_images:
                dest_path = os.path.join(folder, 'melanoma', image)
            else:
                dest_path = os.path.join(folder, 'no_melanoma', image)
            shutil.move(source_path, dest_path)

print("Las imágenes han sido organizadas.")
