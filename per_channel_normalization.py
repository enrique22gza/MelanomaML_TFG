#NORMALIZACION
#PER-CHANNEL NORMALIZATION
import os
import cv2
import numpy as np
class PerChannelNormalizer:
   def __init__(self):
       self.mean_per_channel = None
       self.std_per_channel = None
   def calculate_stats(self, dataset):
       # Calcula la media y desviación estándar por canal
       self.mean_per_channel = np.mean(dataset, axis=(0, 1, 2))
       self.std_per_channel = np.std(dataset, axis=(0, 1, 2))
       # Debug: imprime los valores calculados para verificar
       print("Mean per channel:", self.mean_per_channel)
       print("Std per channel:", self.std_per_channel)
   def per_channel_normalize(self, image):
       if self.mean_per_channel is None or self.std_per_channel is None:
           raise ValueError("Statistics not calculated. Call calculate_stats with the dataset first.")
       # Normaliza la imagen
       return (image - self.mean_per_channel) / self.std_per_channel
       
def load_dataset(folder_path):
   dataset = []
   for filename in os.listdir(folder_path):
       if filename.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
           img_path = os.path.join(folder_path, filename)
           img = cv2.imread(img_path)
           if img is not None:
               dataset.append(img)
           else:
               print(f"Failed to load image: {filename}")
   return np.array(dataset)
def save_normalized_images(normalized_dataset, output_folder):
   if not os.path.exists(output_folder):
       os.makedirs(output_folder)
   for i, image in enumerate(normalized_dataset):
       # Reescala los valores a [0, 255] y ajusta el tipo de dato
       image_rescaled = 255 * ((image - image.min()) / (image.max() - image.min()))
       output_path = os.path.join(output_folder, f"normalized_image_{i}.jpg")
       cv2.imwrite(output_path, image_rescaled.astype(np.uint8))
# Define los paths de entrada y salida
input_folder = ''
output_folder = ''
# Carga el dataset de imágenes
dataset = load_dataset(input_folder)
normalizer = PerChannelNormalizer()
# Calcula las estadísticas del dataset
normalizer.calculate_stats(dataset)
# Normaliza cada imagen en el dataset
normalized_dataset = [normalizer.per_channel_normalize(image) for image in dataset]
# Guarda las imágenes normalizadas
save_normalized_images(normalized_dataset, output_folder)
 
mean_normalized = np.mean(normalized_dataset, axis=(0, 1, 2))
std_normalized = np.std(normalized_dataset, axis=(0, 1, 2))

print("Media normalizada por canal:", mean_normalized)
print("Desviación estándar normalizada por canal:", std_normalized)
