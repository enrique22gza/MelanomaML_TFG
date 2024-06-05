#DIVIDIR LOS DATOS EN TRAIN, VAL , TEST [CNNs]
import os
import numpy as np
import shutil

def split_data(source_dir, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15):
    # Asegurarte de que los directorios existan
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Obtener todos los archivos de la carpeta fuente
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    np.random.shuffle(all_files)  # Mezclar aleatoriamente los archivos

    # Calcular los puntos de corte para cada conjunto
    total_files = len(all_files)
    train_end = int(train_size * total_files)
    val_end = train_end + int(val_size * total_files)

    # Dividir los archivos
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # Función para copiar archivos
    def copy_files(files, destination):
        for file in files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(destination, file))

    # Copiar archivos a sus respectivos directorios
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

    print(f'Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}')


source_folder = ''
train_folder = ''
val_folder = ''
test_folder = ''

split_data(source_folder, train_folder, val_folder, test_folder)
