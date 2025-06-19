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



# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de uso de la función
    import os
    import sys
    sys.path.append("/home/guest/code")
    print(sys.path)

    import matplotlib.pyplot as plt
    from DataAnalyzer import DataAnalyzer

    ruta = "/home/guest/work/Datasets/picai_folds/picai_images_fold0/10000"
    imgname = "10000_1000000_t2w.mha"

    input_path = os.path.join(ruta, imgname)  # Cambia por tu ruta
    output_path = "./imagen_preprocesada.nii.gz"
    
    # Cargar la imagen
    img = sitk.ReadImage(input_path)
    img_array = sitk.GetArrayFromImage(img)
    # # Convertir la imagen
    # img_array = convert(img_array, 0, 255, target_type=np.uint8)
    # # Guardar la imagen convertida
    # img_out = sitk.GetImageFromArray(img_array)
    # img_out.CopyInformation(img)

    # Normalizar la imagen usando Z-score
    # img_out = sitk_minmax_normalize(img, output_pixel_type=sitk.sitkFloat32)

    # Example usage:
    
    img_out = normalize_image(img, method='minmax')
    
    # Guardar la imagen procesada
    sitk.WriteImage(img_out, output_path)

    # Mostrar la imagen procesada
    da = DataAnalyzer(".")
    da.show_image(input_path, output_path, save="test.png")

    da.image_intensity_histogram(input_path, plot=True, save="histogram_1.png")
    da.image_intensity_histogram(output_path, plot=True, save="histogram_2.png")