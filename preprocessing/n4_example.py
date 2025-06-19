import SimpleITK as sitk
import numpy as np

def preprocess_mri_image(image_path, normalization_method='zscore', 
                        apply_n4_correction=True, mask_path=None,
                        shrink_factor=4, convergence_tolerance=0.001,
                        max_iterations=[50, 50, 50, 50]):
    """
    Preprocesa una imagen de resonancia magnética aplicando corrección N4ITK 
    y normalización de intensidades.
    
    Parámetros:
    -----------
    image_path : str
        Ruta al archivo de imagen (formato compatible con SimpleITK)
    normalization_method : str, opcional
        Método de normalización ('zscore' o 'minmax'). Por defecto 'zscore'
    apply_n4_correction : bool, opcional
        Si aplicar corrección de campo de sesgo N4ITK. Por defecto True
    mask_path : str, opcional
        Ruta a una máscara binaria. Si None, se crea automáticamente
    shrink_factor : int, opcional
        Factor de reducción para acelerar N4ITK. Por defecto 4
    convergence_tolerance : float, opcional
        Tolerancia de convergencia para N4ITK. Por defecto 0.001
    max_iterations : list, opcional
        Número máximo de iteraciones por nivel. Por defecto [50, 50, 50, 50]
    
    Retorna:
    --------
    sitk.Image
        Imagen preprocesada
    """
    
    # Cargar la imagen
    try:
        image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        print(f"Imagen cargada exitosamente: {image.GetSize()}")
    except Exception as e:
        raise ValueError(f"Error al cargar la imagen: {e}")
    
    # Aplicar corrección de campo de sesgo N4ITK
    if apply_n4_correction:
        print("Aplicando corrección N4ITK...")
        
        # Cargar o crear máscara
        if mask_path:
            try:
                mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
                print("Máscara personalizada cargada")
            except Exception as e:
                print(f"Error al cargar máscara: {e}. Creando máscara automática...")
                mask = _create_automatic_mask(image)
        else:
            mask = _create_automatic_mask(image)
        
        # Configurar filtro N4ITK
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(max_iterations)
        corrector.SetConvergenceThreshold(convergence_tolerance)
        
        # Aplicar corrección con factor de reducción para acelerar procesamiento
        if shrink_factor > 1:
            # Reducir imagen y máscara
            shrinker = sitk.ShrinkImageFilter()
            shrinker.SetShrinkFactors([shrink_factor] * image.GetDimension())
            
            shrunk_image = shrinker.Execute(image)
            shrunk_mask = shrinker.Execute(mask)
            
            # Aplicar N4 en imagen reducida
            log_bias_field = corrector.Execute(shrunk_image, shrunk_mask)
            
            # Expandir campo de sesgo a tamaño original
            #expander = sitk.BSplineTransformInitializerImageFilter()
            log_bias_field = sitk.Resample(log_bias_field, image, 
                                         sitk.Transform(), 
                                         sitk.sitkLinear, 
                                         0.0, 
                                         log_bias_field.GetPixelID())
        else:
            # Aplicar N4 directamente
            log_bias_field = corrector.Execute(image, mask)
        
        # Aplicar corrección
        bias_field = sitk.Exp(log_bias_field)
        corrected_image = image / bias_field
        
        print("Corrección N4ITK completada")
    else:
        corrected_image = image
        print("Saltando corrección N4ITK")
    
    # Normalización de intensidades
    print(f"Aplicando normalización: {normalization_method}")
    normalized_image = _normalize_intensities(corrected_image, 
                                            method=normalization_method,
                                            mask=mask if apply_n4_correction else None)
    
    print("Preprocesado completado exitosamente")
    return normalized_image


def _create_automatic_mask(image, threshold_method='otsu'):
    """
    Crea una máscara automática para la imagen.
    
    Parámetros:
    -----------
    image : sitk.Image
        Imagen de entrada
    threshold_method : str
        Método de umbralización ('otsu' o 'percentile')
    
    Retorna:
    --------
    sitk.Image
        Máscara binaria
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


def _normalize_intensities(image, method='zscore', mask=None):
    """
    Normaliza las intensidades de la imagen.
    
    Parámetros:
    -----------
    image : sitk.Image
        Imagen a normalizar
    method : str
        Método de normalización ('zscore' o 'minmax')
    mask : sitk.Image, opcional
        Máscara para calcular estadísticas solo en región de interés
    
    Retorna:
    --------
    sitk.Image
        Imagen normalizada
    """
    # Convertir a array numpy
    image_array = sitk.GetArrayFromImage(image)
    
    if mask is not None:
        mask_array = sitk.GetArrayFromImage(mask)
        # Calcular estadísticas solo en la región enmascarada
        masked_values = image_array[mask_array > 0]
    else:
        # Usar todos los valores no cero
        masked_values = image_array[image_array > 0]
    
    if method.lower() == 'zscore':
        # Normalización Z-score (media=0, std=1)
        mean_val = np.mean(masked_values)
        std_val = np.std(masked_values)
        
        if std_val == 0:
            print("Advertencia: Desviación estándar es 0. No se puede aplicar Z-score.")
            normalized_array = image_array
        else:
            normalized_array = (image_array - mean_val) / std_val
            
    elif method.lower() == 'minmax':
        # Normalización min-max a rango [0, 1]
        min_val = np.min(masked_values)
        max_val = np.max(masked_values)
        
        if max_val == min_val:
            print("Advertencia: Rango de valores es 0. No se puede aplicar normalización min-max.")
            normalized_array = image_array
        else:
            normalized_array = (image_array - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Método de normalización no reconocido: {method}")
    
    # Convertir de vuelta a imagen SimpleITK
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    
    return normalized_image


def save_preprocessed_image(image, output_path):
    """
    Guarda la imagen preprocesada.
    
    Parámetros:
    -----------
    image : sitk.Image
        Imagen a guardar
    output_path : str
        Ruta donde guardar la imagen
    """
    try:
        sitk.WriteImage(image, output_path)
        print(f"Imagen guardada en: {output_path}")
    except Exception as e:
        print(f"Error al guardar imagen: {e}")


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso de la función
    import os
    '''
    DATA_ROOT = "/home/guest/work/Datasets"
    labels_orig_root  = os.path.join(DATA_ROOT, "picai_labels-main/csPCa_lesion_delineations/human_expert/original")
    labels_resampled_root = os.path.join(DATA_ROOT, "picai_labels-main/csPCa_lesion_delineations/human_expert/resampled")
    images_root = os.path.join(DATA_ROOT, "picai_images_fold0/")

    # get all dirs in images_root
    dirs = [f.path for f in os.scandir(images_root) if f.is_dir()]
    print(len(dirs))
    
    # get a random directory
    # choose the file ending in _adc.mha
    '''

    ruta = "/home/guest/work/Datasets/picai_images_fold0/10000"

    input_path = os.path.join(ruta, "10000_1000000_adc.mha")  # Cambia por tu ruta
    output_path = "./imagen_preprocesada.nii.gz"
    
    try:
        # Preprocesar imagen con Z-score
        processed_image = preprocess_mri_image(
            image_path=input_path,
            normalization_method='zscore',
            apply_n4_correction=True,
            shrink_factor=4
        )
        
        # Guardar resultado
        save_preprocessed_image(processed_image, output_path)
        
        # También se puede usar normalización min-max
        processed_image_minmax = preprocess_mri_image(
            image_path=input_path,
            normalization_method='minmax',
            apply_n4_correction=True
        )
        
        save_preprocessed_image(processed_image_minmax, "imagen_preprocesada_minmax.nii.gz")
        
    except Exception as e:
        print(f"Error en el preprocesado: {e}")