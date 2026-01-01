import argparse
import os

from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_float

import matplotlib.pyplot as plt

import math
import linalg

import numpy.testing as npt
from numpy.typing import NDArray
import numpy as np

import encoder

import rawpy
import imageio.v3 as iio

from pathlib import Path

import re

"""
    Prints the original images size in number of bytes

    Args:
        image_file_path (str): The file path to original image

    Returns:
        None
"""
def get_image_metadata(image_file_path: str) -> None:

    if not image_file_path:
        print("No image path provided.")
        return

    if not os.path.exists(image_file_path):
        print(f"File not found: {image_file_path}")
        return

    size = os.path.getsize(image_file_path)
    print(f"Original image size: {size} bytes")



"""
    Converts an image to greyscale

    Args:
        image (NDArray[uint8]): A 3d tensor containing each pixel and their corresponding
                                RGB values for image

    Returns:
        NDArray[uint8]: The original 3d image tensor converted to greyscale
"""
def convert_to_greyscale(image_name: str, image: NDArray[np.uint8], image_file_path: str) -> NDArray[np.uint8]:

    image = image_to_array(image_file_path)

    image_matrix = np.asarray(image)
    image_matrix_greyscale = image_matrix.astype(np.float64)

    print("Input image dimensions:")
    print(f" Width: {image_matrix.shape[0]}")
    print(f"Height: {image_matrix.shape[1]}")
    print(f" Depth: {image_matrix.shape[2]}")

    # Note: Even when loading in an already grayscale image, it will still have 3 depth dimensions because 
    # its simpler to keep the RGB structure intact and set all values for the 3 colors to the same to produce grey


    H, W, C = image_matrix.shape # Tensor
    for row in range(H):
        for element in range(W):
            #print(f"Row: {row} Element: {element}")
            
            red = image_matrix[row, element, 0]
            green = image_matrix[row,element, 1]
            blue = image_matrix[row,element, 2]

            # Standard formula 
            gray = 0.299 * red + 0.587 * green + 0.114 * blue

            image_matrix_greyscale[row, element, 0] = gray
            image_matrix_greyscale[row, element, 1] = gray
            image_matrix_greyscale[row, element, 2] = gray

    # Now we must turn the greyscale matrix from 3D into 2D
    new_matrix = np.empty((H, W), dtype=np.uint8)

    for row in range(H):
        for element in range(W):

            new_matrix[row, element] = image_matrix_greyscale[row, element, 0]

    image_output_directory = "../bin"
    output_path = os.path.join(image_output_directory, f"{image_name}_greyscale.jpg")

    plt.imsave(output_path, new_matrix, cmap="gray")

    return new_matrix


def image_to_array(image_file_path: str) -> NDArray[np.uint8]:
    
    p = Path(image_file_path)
    extension = p.suffix

    # Raw format
    if extension in ('.tiff', '.RAF', '.CR2', '.CR3', '.NEF', '.NRW', '.ORF', '.RW2'):
        with rawpy.imread(image_file_path) as raw:
            rgb = raw.postprocess()
            return rgb.astype(np.float32)

    elif extension in ('.jpg', '.jpeg', '.png', '.GIF', '.webp', '.bmp'):
        return imread(image_file_path)



# MAKE SURE MATRIX IS UINT8
def compress(image_name: str, k: int, svd: tuple, matrix: NDArray[np.uint8]) -> NDArray[np.float64]:
    U, sigma, VT = svd    

    U_k  = U[:, :k]
    VT_k = VT[:k, :]

    Sigma_k = sigma[:k, :k]

    A_k = U_k @ Sigma_k @ VT_k
    A_k = np.clip(A_k, 0, 255)
    A_k = A_k.astype(np.uint8)

    out_dir = "../bin"
    out_path = os.path.join(out_dir, f"{image_name}_compressedk{k}.jpg")
    plt.imsave(out_path, A_k, cmap="gray")

    size = os.path.getsize(out_path)
    print(f"Compressed image size: {size} bytes")



    H, W = matrix.shape
    metadata = (U_k, Sigma_k, VT_k, H, W, A_k.dtype)
    encoder.write_output(image_name, metadata)

    return A_k



"""
    Calucaltes Costs

    Args:
        k      (int):
        height (int):
        width  (int):

    Returns:
        None
"""
def calculate_costs(k: int, height: int, width: int) -> None:

    raw_cost = height * width
    svd_cost = height*k + width*k + k

    print("Raw entries:", raw_cost)
    print("SVD entries:", svd_cost)

    print(f"Compression ratio: {raw_cost/svd_cost}")



"""
    Parses a image file path to extract the image name and extension(file type)

    Args:
        image_file_path (str): Path to image

    Returns:
        tuple(str,str): A tuple containing first the image name, then second the extension 
"""
def parse_image_path(image_file_path: str) -> tuple[str, str]:

    filename = os.path.basename(image_file_path)
    name, extension = os.path.splitext(filename)         
    return name, extension
