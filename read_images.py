import numpy as np
from pathlib import Path
from PIL import Image
from natsort import natsorted
from astropy.io import fits
from scipy.ndimage import median_filter

def load_and_sort_images(directory):
    # Create a Path object for the directory
    dir_path = Path(directory)

    # Find all .jpg files and sort them naturally
    jpg_files = natsorted(dir_path.rglob('*.tiff'))

    # Load and convert each image to a numpy array
    images = [np.array(Image.open(file)) for file in jpg_files]

    # Check if images are not empty and have consistent dimensions
    if not images or len(set(img.shape for img in images)) != 1:
        raise ValueError("Images are either empty or have inconsistent dimensions")

    # Stack all images into a multi-dimensional numpy array
    images_stack = np.stack(images)

    # Transpose to get dimensions as (n_images, channels, x, y)
    if images_stack.ndim == 4:  # Assuming images are colored (RGB)
        images_stack = images_stack.transpose(0, 3, 1, 2)

    return images_stack


def save_numpy_as_fits(array, output_directory="./data/scratch/",
                       filename="default"):
    """
    Saves each slice of a numpy array as a separate .fits file.

    Parameters:
    - array: numpy array where each slice along the first axis is an image.
    - output_directory: Directory where the .fits files will be saved.
    - filename: Base name for the output files. The function will append a number for each slice.
    """


    # Create a PrimaryHDU object from the numpy array slice
    hdu = fits.PrimaryHDU(array)
    hdul = fits.HDUList([hdu])

    # Generate the output file name
    filename = Path(output_directory) / f"{filename}.fits"

    # Write the fits file
    hdul.writeto(filename, overwrite=True)

if __name__ == "__main__":
    DIR = Path("./data/T7_2022-12-05_16-24-10/R/")
    imgs = load_and_sort_images(DIR)

    imgs = imgs.astype(np.float32)

    imgs_medsub = imgs - np.median(imgs, axis=0)

    imgs_medfilt_sub = imgs - median_filter(imgs, size=(20, 1, 1, 1), mode="mirror", origin=(-10, 0, 0, 0))

    save_numpy_as_fits(imgs, filename="test")
    save_numpy_as_fits(imgs_medsub, filename="imgs_medsub")
    save_numpy_as_fits(imgs_medfilt_sub, filename="imgs_medfilt_sub")