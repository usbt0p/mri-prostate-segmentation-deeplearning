import numpy as np
import SimpleITK as sitk

def normalize_image(image: sitk.Image, method: str = 'zscore') -> sitk.Image:
    """
    Normalize a SimpleITK image using Z-score or Min-Max normalization.

    Parameters:
        image (sitk.Image): Input image.
        method (str): Normalization method, 'zscore' or 'minmax'.

    Returns:
        sitk.Image: Normalized image.
    """
    array = sitk.GetArrayFromImage(image)  # shape: [slices, height, width]
    non_zero = array[array > 0]  # Avoid background
    
    if method == 'zscore':
        mean = non_zero.mean()
        std = non_zero.std()
        norm_array = (array - mean) / std
    elif method == 'minmax':
        min_val = non_zero.min()
        max_val = non_zero.max()
        norm_array = (array - min_val) / (max_val - min_val)
    else:
        raise ValueError("Normalization method must be 'zscore' or 'minmax'")

    norm_array[array == 0] = 0  # Optional: preserve background as 0

    norm_image = sitk.GetImageFromArray(norm_array)
    norm_image.CopyInformation(image)

    return norm_image

def create_automatic_mask(image, threshold_method='otsu'):
    """
    Crea una máscara automática para la imagen.
    """
    if threshold_method == 'otsu':
        # Usar umbralización Otsu
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        mask = otsu_filter.Execute(image)
    else:
        # Usar umbralización por percentil
        image_array = sitk.GetArrayFromImage(image)
        threshold = np.percentile(image_array[image_array > 0], 10)
        mask = sitk.BinaryThreshold(image, lowerThreshold=threshold, 
                                  upperThreshold=image_array.max(),
                                  insideValue=1, outsideValue=0)
    
    # Operaciones morfológicas para limpiar la máscara
    mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
    mask = sitk.BinaryFillhole(mask)
    
    return sitk.Cast(mask, sitk.sitkUInt8)

def n4_correction(image : sitk.Image, 
                  mask_path=None, # TODO pasar una máscara de ROI
                  shrink_factor=4, 
                  convergence_tolerance=0.001,
                  max_iterations=[50, 50, 50, 50]):        

    
    # Aplicar corrección de campo de sesgo N4ITK
    
    # TODO ver si puede pasarle una máscara de ROI de la glándula principal
    # Cargar o crear máscara
    # if mask_path:
    #     try:
    #         mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    #         print("Máscara personalizada cargada")
    #     except Exception as e:
    #         print(f"Error al cargar máscara: {e}. Creando máscara automática...")
    #         mask = _create_automatic_mask(image)
    # else:
    #     mask = _create_automatic_mask(image)
    
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Configurar filtro N4ITK
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(max_iterations)
    #corrector.SetConvergenceThreshold(convergence_tolerance)
    
    # Aplicar corrección con factor de reducción para acelerar procesamiento
    if shrink_factor > 1:
        # Reducir imagen y máscara
        shrinker = sitk.ShrinkImageFilter()
        shrinker.SetShrinkFactors([shrink_factor] * image.GetDimension())
        
        shrunk_image = shrinker.Execute(image)
        #shrunk_mask = shrinker.Execute(mask)
        
        # Aplicar N4 en imagen reducida
        log_bias_field = corrector.Execute(shrunk_image)#, shrunk_mask)
        
        # Expandir campo de sesgo a tamaño original
        #expander = sitk.BSplineTransformInitializerImageFilter()
        log_bias_field = sitk.Resample(log_bias_field, image, 
                                        sitk.Transform(), 
                                        sitk.sitkLinear, 
                                        0.0, 
                                        log_bias_field.GetPixelID())
    else:
        # Aplicar N4 directamente
        log_bias_field = corrector.Execute(image)#, mask)
    
    # Aplicar corrección
    bias_field = sitk.Exp(log_bias_field)
    corrected_image = image / bias_field
    print("Corrección N4ITK completada")
    
    return corrected_image

import SimpleITK as sitk

def n4_bias_field_correction(
    image: sitk.Image,
    shrink_factor: int = 4,
    num_iterations: int = 50,
    num_fitting_levels: int = 4
) -> sitk.Image:
    """
    Aplica corrección N4 del campo de sesgo a toda la imagen sin usar máscara.

    Parámetros:
        image: imagen de entrada (Float32 o Float64).
        shrink_factor: factor para reducir resolución (por defecto 4).
        num_iterations: iteraciones por nivel de ajuste.
        num_fitting_levels: niveles en la jerarquía multi‐escala.

    Retorna:
        Imagen con corrección del bias field aplicada.
    """
    # Asegurar tipo float
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Crear máscara si no se suministra
    #if mask is not None:
        # TODO see if this is necessary
        #mask = sitk.OtsuThreshold(image, 0, 1, 200)
        
    # Reducir resolución para acelerar cálculo
    if shrink_factor > 1:
        small = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    else:
        small = image

    # Configurar filtro N4
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iterations] * num_fitting_levels)

    # Ejecutar corrección sin máscara: se usa toda la imagen
    corrected_small = corrector.Execute(small)

    # Reconstruir campo de sesgo logaritmico en resolución original
    log_bias = corrector.GetLogBiasFieldAsImage(image)

    # Aplicar corrección completa
    corrected = image / sitk.Exp(log_bias)

    # Copiar información de espacialidad
    corrected.CopyInformation(image)

    return corrected, log_bias


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso de la función
    import os
    import sys


    import matplotlib.pyplot as plt
    from DataAnalyzer import DataAnalyzer

    ruta = "/home/guest/work/Datasets/picai_folds/picai_images_fold0/10000"
    imgname = "10000_1000000_t2w.mha"

    input_path = os.path.join(ruta, imgname)  # Cambia por tu ruta
    output_path = "./imagen_preprocesada.nii.gz"
    
    # Cargar la imagen
    img = sitk.ReadImage(input_path)
    img_array = sitk.GetArrayFromImage(img)

    # Example usage of norm:
    #img_out = normalize_image(img, method='minmax')


    # example usage of n4_correction:
    img_out, log_bias = n4_bias_field_correction(img)

    # example usage of automatic mask:
    #img_out = create_automatic_mask(img)
    
    # TODO este método parece devolver mierda, revisar. Mira en los papers si ellos lo usa
    # TODO: mirar si el metodo es necesario, probar n4 sin
    # si lo es incorporar a n4, si no quitarlo
    # luego probar n4, y luego pasar al registro, roi y resampleo. 
    # estudiar si hacer interpolacion de contornos d las mascaras?
    
    # Guardar la imagen procesada
    sitk.WriteImage(img_out, output_path)
    
    logb_path = "./log_bias.nii.gz"
    sitk.WriteImage(log_bias, logb_path)


    # Mostrar la imagen procesada
    da = DataAnalyzer(".")
    da.show_image(input_path, logb_path, output_path, save="test.png")

    da.image_intensity_histogram(input_path, plot=True, save="histogram_1.png")
    da.image_intensity_histogram(output_path, plot=True, save="histogram_2.png")