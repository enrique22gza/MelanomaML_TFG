#HAIR REMOVAL- DULLRAZOR (+ APPLICATION OF A GAUSSIAN FILTER)

import os
import cv2

def process_image(input_path, output_path):
    # Read image
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"No se pudo leer la imagen en {input_path}")
        return
    
    
    # Gray scale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Black hat filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) 
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # Gaussian filter
   
    
    # Binary thresholding (MASK)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    
    # Replace pixels of the mask
    clean_image = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)   
    
    # Write the processed image to the output folder
    filename = os.path.basename(input_path)
    output_file = os.path.join(output_path, filename)
    cv2.imwrite(output_file, clean_image)

def process_images_in_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png','.bmp')):  # Filter only image files
            input_path = os.path.join(input_folder, filename)
            process_image(input_path, output_folder)

# Input and output folders
input_folder = ''
output_folder = ''

# Process images in the input folder and save the results in the output folder
process_images_in_folder(input_folder, output_folder)
