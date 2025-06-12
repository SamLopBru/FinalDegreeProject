import cv2
import numpy as np
from scipy.signal import wiener

def aplicarClahe(final_image, clipLimit: float=1.0, tileSize: tuple=(10,10) ):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileSize)
    image = (((final_image - np.min(final_image)) / (np.max(final_image) - np.min(final_image)))*255).astype(np.uint8)
    final_image = clahe.apply(image)
    return final_image

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

def procesar_slice(i):
    i = aplicarAWA(i)
    i = aplicarFiltros(i)
    i = aplicarClahe(i)
    return i

