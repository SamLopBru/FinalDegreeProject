import cv2
import numpy as np
import pydicom as py
from scipy.signal import wiener
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def aplicarClahe(final_image, clipLimit: float=1.0, tileSize: tuple=(10,10) ):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileSize)
    image = (((final_image - np.min(final_image)) / (np.max(final_image) - np.min(final_image)))*255).astype(np.uint8)
    return clahe.apply(image)

def aplicarAWA(img):
    """
    Apply Anscombe transform, Wiener filter and Anscombe inverse transform (Yousefi y cols. (2018))
    
    PARAMS:
    img --> pixel array of an image
    """
    img_anscombe = anscombe_transform(img)

    img_wiener = wiener(img_anscombe, (5, 5))
    img_wiener = np.nan_to_num(img_wiener)
    
    img_final = anscombe_inverse(img_wiener)

    img_final = cv2.normalize(img_final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_final

def aplicarFiltros(image):
    
    _,mask = cv2.threshold(image, 240,255, cv2.THRESH_TRIANGLE)
    contour = cv2.Canny(mask,100,200)

    kernel = np.ones((15,15),np.uint8)
    dilated = cv2.dilate(contour, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)  #identify largest contour

    mask_cleaned = np.zeros_like(image) 

    cv2.drawContours(mask_cleaned, [largest_contour], -1, 255, thickness=cv2.FILLED) 

    clean_edges = cv2.bitwise_and(dilated, mask_cleaned)
    _,clean_edges_inverse = cv2.threshold(clean_edges,0,255,cv2.THRESH_BINARY_INV)

    final_result = cv2.bitwise_and(image,clean_edges_inverse)

    return final_result

def anscombe_transform(img):
    return 2 * np.sqrt(img + 3/8)

def anscombe_inverse(y):
    return (y / 2) ** 2 - 1/8

def process_single_slice(args):
    slice, output_path, num_slice = args
    try:
        awa = aplicarAWA(slice)
        filtered = aplicarFiltros(awa)
        clahe = aplicarClahe(filtered)
        
        final_path = os.path.join(output_path, f"{num_slice}.png")
        Image.fromarray(clahe).save(final_path)
        return True
    except Exception as e:
        print(f"Error procesando slice {num_slice}: {str(e)}")
        return False

def saveSlices(dicom_path, output_path):
    """This function makes the preprocessing and then the PNG of all the images with multiprocessing
    Args:
    dicom_path: dicom image to preprocess and transform to png
    output_path: directory when the transformed images are going to be
    """
    os.makedirs(output_path, exist_ok=True)
    
    pixel_array = py.dcmread(dicom_path, force=True).pixel_array

    num_slices = pixel_array.shape[0]
    
    args = [(slice, output_path, i) for i, slice in enumerate(pixel_array)]
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_slice, args))
    
    return results, num_slices
    
def toPNG(main_folder,output_folder):

    all_num_slices = []

    lista_pacientes = os.listdir(main_folder)

    for paciente in lista_pacientes:
        
        folder_paciente = os.path.join(main_folder, paciente)

        folder_salida_paciente = os.path.join(output_folder,paciente)    
        try:
            os.mkdir(folder_salida_paciente)
            lista_carpetas_intermedias = os.listdir(folder_paciente)

            for carpeta in lista_carpetas_intermedias:

                folder_vistas = os.path.join(folder_paciente,carpeta)

                lista_vistas = os.listdir(folder_vistas)
                
                for vista in lista_vistas:
                    
                    folder_image = os.path.join(folder_vistas,vista)
                    
                    folder_salida_vista = os.path.join(folder_salida_paciente,vista)
                    
                    os.mkdir(folder_salida_vista)
                    
                    image = os.path.join(folder_image, os.listdir(folder_image)[0])
                    _,num_slice = saveSlices(image, folder_salida_vista)
                    all_num_slices.append(num_slice)

        except FileExistsError:
            continue

    return max(all_num_slices), min(all_num_slices)


                   
                

if __name__=='__main__':
    main_folder = r''

    output_folder = r''

    max_slices, min_slices = toPNG(main_folder, output_folder)




