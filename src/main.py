import compress
import linalg
import argparse
from skimage.io import imread
import numpy as np

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("-ip", "--imagePath", type=str, nargs='?', const='', help='Path to image to be compressed')
    parser.add_argument("-k", "--rank", type=int, help='Number of singular values to use in compression')


    args = parser.parse_args()

    image_name, extension = compress.parse_image_path(args.imagePath)

    compress.get_image_metadata(image_file_path=args.imagePath)

    matrix = compress.convert_to_greyscale(image_name, args.imagePath, args.imagePath)
    
    svd = linalg.svd(matrix)
    print(f"Frobenius Error between A and A_k: {linalg.frobenius_error(svd[1], args.rank)}")

    A = compress.compress(image_name, args.rank, svd, matrix)

    height, width = matrix.shape

    compress.calculate_costs(args.rank, height, width)


if __name__ == "__main__":
    main()
