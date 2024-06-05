import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns

# Rutas
ruta_imagenes = ''
ruta_mascaras_verdaderas = ''
ruta_mascaras_otsu = ''

# Función para cargar imágenes y máscaras
def cargar_imagenes_y_mascaras(lista_archivos, ruta_imagenes, ruta_mascaras_verdaderas, ruta_mascaras_otsu):
    imagenes = []
    mascaras_verdaderas = []
    mascaras_otsu = []
    
    for archivo in lista_archivos:
        imagen = cv2.imread(os.path.join(ruta_imagenes, archivo))
        mascara_verdadera = cv2.imread(os.path.join(ruta_mascaras_verdaderas, archivo.replace('.png', '_lesion.png')), cv2.IMREAD_GRAYSCALE)
        mascara_otsu = cv2.imread(os.path.join(ruta_mascaras_otsu, archivo.replace('.png', '_mascara_otsu_invertida.png')), cv2.IMREAD_GRAYSCALE)
        
        imagenes.append(imagen)
        mascaras_verdaderas.append(mascara_verdadera)
        mascaras_otsu.append(mascara_otsu)
    
    return np.array(imagenes), np.array(mascaras_verdaderas), np.array(mascaras_otsu)

# Obtiene la lista de archivos de imágenes
archivos_imagenes = [f for f in os.listdir(ruta_imagenes) if f.endswith('.png')]

# Cargar las imágenes y máscaras
imagenes, mascaras_verdaderas, mascaras_otsu = cargar_imagenes_y_mascaras(archivos_imagenes, ruta_imagenes, ruta_mascaras_verdaderas, ruta_mascaras_otsu)

# Convertir máscaras de 0-255 a 0-1
mascaras_verdaderas_bin = (mascaras_verdaderas > 0).astype(np.uint8)
mascaras_otsu_bin = (mascaras_otsu > 0).astype(np.uint8)

# Flatten the arrays for metric calculations
mascaras_verdaderas_flat = mascaras_verdaderas_bin.flatten()
mascaras_otsu_flat = mascaras_otsu_bin.flatten()

# Cálculo de las métricas
precision = precision_score(mascaras_verdaderas_flat, mascaras_otsu_flat)
recall = recall_score(mascaras_verdaderas_flat, mascaras_otsu_flat)
f1 = f1_score(mascaras_verdaderas_flat, mascaras_otsu_flat)
iou = jaccard_score(mascaras_verdaderas_flat, mascaras_otsu_flat)
tn, fp, fn, tp = confusion_matrix(mascaras_verdaderas_flat, mascaras_otsu_flat).ravel()
specificity = tn / (tn + fp)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"IoU: {iou}")
print(f"Specificity: {specificity}")

# Curva ROC y AUC
fpr, tpr, _ = roc_curve(mascaras_verdaderas_flat, mascaras_otsu_flat)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



# Matriz de Confusión
conf_matrix = confusion_matrix(mascaras_verdaderas_flat, mascaras_otsu_flat)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Visualización de imágenes, máscaras verdaderas y máscaras predichas
for i in range(5):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(mascaras_verdaderas_bin[i], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(mascaras_otsu_bin[i], cmap='gray')
    plt.axis('off')
    plt.show()
