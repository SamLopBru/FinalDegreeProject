import os

def select_slices(patient_dir, num_slices=27):
    """
    Select num_slices with the following criteria:
    - Firt select does slices with rest 0 when divided by 3
    - Second select the slices +2 of the group before
    - Third select the slices +1 of the first group 
    - Take the num_slices as result

    :param patient_dir: folder path with slices.
    :param num_slices: max number of slices to select (default 27)
    :return: Lista con las rutas de los slices seleccionados.
    """
    slices = sorted([int(f.split(".")[0]) for f in os.listdir(patient_dir) if f.endswith('.png')])

    if not slices:
        raise ValueError(f"No se encontraron imágenes en {patient_dir}")

    total_slices = len(slices)

    if total_slices >= 100 and total_slices < 110:
        slices = slices[3:total_slices-3]
    elif total_slices >=  110 and total_slices<120:
        slices = slices[6:total_slices-6]
    
    first_turn = [str(s)+'.png' for s in slices if int(s) % 3 == 0]
    
    second_turn = []
    for s in slices:
        if int(s) % 3 == 0:
            if int(s)+2 < total_slices:
                second_turn.append(str(s+2)+'.png')
    
    third_turn = []
    for s in slices:
        if int(s) % 3 == 0:
            if int(s)+1 < total_slices:
                third_turn.append(str(s+1)+'.png')
    
    
    selected_slices = first_turn + second_turn + third_turn
    selected_slices = selected_slices[:num_slices]
    selected_slices = [sorted([int(f.split(".")[0]) for f in selected_slices if f.endswith('.png')])][0]
    selected_slices = [str(s)+'.png' for s in selected_slices]

    return [os.path.join(patient_dir,s) for s in selected_slices]

def eliminarSlices(carpeta_principal):

    lista_CarpetaPacientes = os.listdir(carpeta_principal)

    for paciente in lista_CarpetaPacientes:

        carpetas_Intermedias = os.path.join(carpeta_principal,paciente)
        lista_CarpetasIntermedias = os.listdir(carpetas_Intermedias)

        for carpetaIntermedia in lista_CarpetasIntermedias:

            carpeta_Img = os.path.join(carpetas_Intermedias, carpetaIntermedia)
            all_images = os.listdir(carpeta_Img)
            selected_slices = select_slices(carpeta_Img)
            for img in all_images:
                img = os.path.join(carpeta_Img,img)
                
                if img not in selected_slices:
                    os.remove(img)
            slices = []     
            if len(os.listdir(carpeta_Img)) != 27:
                
                for i in selected_slices:
                    i = i.split("\\")
                    slices.append(i[-1])
                print(f"Selected{slices}")
                print(carpeta_Img)
                raise ValueError("QUE COJONES")
                  
        

if __name__=='__main__':
    carpeta_principal = r'E:\test+valid_png_original'
    eliminarSlices(carpeta_principal)
