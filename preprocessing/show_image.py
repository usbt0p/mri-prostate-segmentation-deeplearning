import SimpleITK as sitk
import matplotlib.pyplot as plt

def show_mha_image(image_path, save_path=None):
    """
    Opens and displays an image in .mha format. Optionally saves the image as a PNG.

    Parameters:
    -----------
    image_path : str
        Path to the .mha image file.
    save_path : str, optional
        Path to save the displayed image (e.g., as a .png file). If None, does not save.
    """
    try:
        # Load the image using SimpleITK
        image = sitk.ReadImage(image_path)
        
        # Convert the image to a numpy array for visualization
        image_array = sitk.GetArrayViewFromImage(image)

        # Take a slice of the image (e.g., the middle slice)
        if image_array.ndim == 3:
            image_array = image_array[image_array.shape[0] // 2, :, :]
        elif image_array.ndim == 2:
            image_array = image_array[:, :]
        
        # Display the image using matplotlib
        plt.imshow(image_array, cmap='gray')
        plt.title(".mha Image")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()
    except Exception as e:
        print(f"Error opening the image: {e}")

if __name__ == "__main__":
    route = "/home/guest/code/imagen_preprocesada_minmax.nii.gz"

    show_mha_image(route, save_path=".")
    