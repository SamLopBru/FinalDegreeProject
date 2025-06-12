import os
from PIL import Image
from multiprocessing import Pool, cpu_count

def reduce_image(path):
    try:
        imagen = Image.open(path)
        nuevo_tamano = (256, 256)
        imagen_redimensionada = imagen.resize(nuevo_tamano, Image.LANCZOS)
        imagen_redimensionada.save(path, optimize=True)
    except Exception as e:
        print(f"Error procesando {path}: {e}")

def get_image_paths(base_folder):
    rutas = []
    for paciente in os.listdir(base_folder):
        carpetas_paciente = os.path.join(base_folder, paciente)
        if not os.path.isdir(carpetas_paciente):
            continue
        for intermedia in os.listdir(carpetas_paciente):
            carpeta_imagenes = os.path.join(carpetas_paciente, intermedia)
            if not os.path.isdir(carpeta_imagenes):
                continue
            for img in os.listdir(carpeta_imagenes):
                ruta_imagen = os.path.join(carpeta_imagenes, img)
                if ruta_imagen.lower().endswith(('.png', '.jpg', '.jpeg')):  
                    rutas.append(ruta_imagen)
    return rutas

if __name__ == '__main__':
    carpeta_principal = r'C:\Users\samue\Documents\UNI\TFG\test+valid_png'
    rutas_imagenes = get_image_paths(carpeta_principal)

    # Usar todos los núcleos disponibles para procesar más rápido
    with Pool(cpu_count()) as pool:
        pool.map(reduce_image, rutas_imagenes)

