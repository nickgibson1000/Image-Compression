import compress
import linalg
import argparse
from skimage.io import imread
import numpy as np

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("-ip", "--imagePath", type=str, nargs='?', const='', help='Path to image to be compressed')

    args = parser.parse_args()

    compress.print_image_size(image_file_path=args.imagePath)

    matrix = compress.convert_to_greyscale(imread(args.imagePath))
    
    svd = linalg.svd(matrix)
    A = compress.compress(20, svd)

    print("Original:", matrix.shape)
    print("Reconstructed:", A.shape)

    # Store off the orirignal and reconstructed arrays
    array_save_directory = "../bin/npz"
    np.savez(array_save_directory, "original_greyscale.npz", matrix)
    np.savez(array_save_directory, "compressed_greyscale.npz", A)



if __name__ == "__main__":
    main()
