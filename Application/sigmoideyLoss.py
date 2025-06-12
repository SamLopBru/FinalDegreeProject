import numpy as np
import cv2
import math


def apply_window_level(image, window:int = None, level:int = None):
    """
    Aply windowing (window) and windowing level (level) to the image
    window: intensity range
    level: center of the intensity range
    """
    max_val = image.max()
    min_val = image.min()
    
    if window is None:
        window = max_val - min_val
    
    if level is None:
        level = (max_val + min_val) / 2
    
    # Superior and inferior limits
    lower_bound = level - (window / 2)
    upper_bound = level + (window / 2)
    
    # Clipping y normalización al rango [0, 255]
    adjusted_image = np.clip(image, lower_bound, upper_bound)

    
    # Normalizar al rango [0,1]
    #windowed = (adjusted_image - lower_bound) / window
    
    return adjusted_image

def apply_mip(tomosynthesis, mode):
    """
    Apply Maximum Intensity Projection (MIP) to the volume
    """

    if mode == "max":
        return np.max(tomosynthesis, axis=0)
    elif mode== "min":
        return np.min(tomosynthesis, axis=0)
    elif mode == "mean":
        return np.mean(tomosynthesis, axis=0)
    else:
        raise ValueError(f"Mode value is incorrect, should be: max, min or mean, got {mode}")

def inverse_log_weight(distance, base:float=10):
    """
    Calculates the weight using an inverse logarithmic function
    Args:
        distance: distance from the center slice
        base: base of the logarithm (controls how fast the weight falls)
    """
    return 1 / (1 + np.log(1 + distance * (base - 1)))

def sigmoid_weight(distance, steepness=2, center=2):
    """
    Calculates the weight using a normalized sigmoid function
    Args:
        distance: distance from the center slice
        steepness: controls the slope of the sigmoid
        center: center point of the transition
    """
    return 1 / (1 + np.exp(steepness * (distance - center)))

def get_sliced_images(image_list, center_slice, num_slices):
    # calculate the start and end indices for slicing
    half_slices = num_slices // 2  
    start_index = max(center_slice - half_slices, 0)  # ensure to not go below 0
    end_index = min(center_slice + half_slices + 1, len(image_list))  # ensure to not go above the length of the list

    
    total_slices = end_index - start_index
    if total_slices < num_slices:
        if start_index == 0:  
            end_index = min(num_slices, len(image_list))
        elif end_index == len(image_list):  
            start_index = max(len(image_list) - num_slices, 0)
    
    sliced_images = image_list[start_index:end_index]
    return sliced_images

def create_synthetic_2d_mip_weighted(tomosynthesis, num_slices, mode:str, window:int=None, level:int=None, 
                                   weight_function='log', weight_params=None,
                                   mip_threshold:float=0.8, center_slice = None):
    """
    Creates a 2D synthetic image by combining MIP and selected weight function.
    
    Parameters:
    tomosynthesis: 3D array with the slices (depth, height, width).
    window: desired intensity range
    level: center of intensity range
    weight_function: 'log' or 'sigmoid'.
    weight_params: dictionary with parameters for the weight function
        for 'log': {'base': value}
        For 'sigmoid': {'steepness': value, 'center': value}
    mip_threshold: threshold for combining MIP with the weighted image (0-1)
    """
    
    if weight_params is None:
        weight_params = {'base': 2} if weight_function == 'log' else {'steepness': 2, 'center': 2}
    
    depth, height, width = tomosynthesis.shape

    tomosynthesis = get_sliced_images(tomosynthesis, center_slice, num_slices)
    
    mip_image = apply_mip(tomosynthesis, mode)
    
    if center_slice is None:
        center_slice = depth // 2
    else:
        center_slice = center_slice
    
    synthetic_image = np.zeros((height, width))
    weight_sum = 0
    
    for i in range(tomosynthesis.shape[0]):
        distance = abs(i - center_slice)
        
        if weight_function == 'log':
            weight = inverse_log_weight(distance, weight_params.get('base', 2))
        else:  # sigmoid
            weight = sigmoid_weight(distance, 
                                 weight_params.get('steepness', 2),
                                 weight_params.get('center', 2))

        synthetic_image += tomosynthesis[i] * weight
        weight_sum += weight
    
    synthetic_image = synthetic_image / weight_sum
    
    mip_image = (mip_image - mip_image.min()) / (mip_image.max() - mip_image.min())
    synthetic_image = (synthetic_image - synthetic_image.min()) / (synthetic_image.max() - synthetic_image.min())
    
    mask = mip_image > mip_threshold
    final_image = np.where(mask, mip_image, synthetic_image)
    
    final_image = apply_window_level(final_image, window, level)
    
    return final_image, mip_image, synthetic_image

def aplicarClahe(final_image, clipLimit: float=1.0, tileSize: tuple=(10,10) ):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileSize)
    image = (((final_image - np.min(final_image)) / (np.max(final_image) - np.min(final_image)))*65535).astype(np.uint16)
    final_image = clahe.apply(image)
    return final_image

def draw_box(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color = None,
    lw=4,
):
    """Draw bounding box on the image"""
    x = min(max(x, 0), image.shape[1] - 1)
    y = min(max(y, 0), image.shape[0] - 1)
    if color is None:
        color = np.max(image)
    if len(image.shape) > 2 and not hasattr(color, "__len__"):
        color = (color,) + (0,) * (image.shape[-1] - 1)
    image[y : y + lw, x : x + width] = color
    image[y + height - lw : y + height, x : x + width] = color
    image[y : y + height, x : x + lw] = color
    image[y : y + height, x + width - lw : x + width] = color
    return image

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

def finalMIP(tomo_slices:list, mode:str, mip_function:str, window:float, level:float, center_slice:int, num_slices:int, clahe:bool=False):

    params = None
    
    if mip_function != "Sigmoid Normalize" and mip_function != "Logaritmic Inverse":
        raise ValueError(f"mip function has two options: 'Sigmoid Normalize' or 'Logaritmic Inverse', got {mip_function}")
    
    if num_slices > len(tomo_slices):
        raise ValueError(f"num_slices cannot be higher than len(tomo_slices), got {len(tomo_slices)}")
    
    if mip_function == "Sigmoid Normalize":
        params = {
        'steepness': 0.7,
        'center': 5
        }
    else:
        params = {
        'base': math.e
        }

    final_image, mip, sigmoid_image = create_synthetic_2d_mip_weighted(
            tomo_slices,
            num_slices,
            mode,
            window = window,
            level = level,
            weight_function = mip_function,
            weight_params = params ,
            mip_threshold = 1, 
            center_slice = center_slice
        )

    final_image = (((final_image - np.min(final_image)) / (np.max(final_image) - np.min(final_image)))*255).astype(np.uint8)
    filtered = aplicarFiltros(final_image)
    if clahe == False:
        filtered = aplicarClahe(filtered)
    
    return filtered

