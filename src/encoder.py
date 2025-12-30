
import numpy as np

"""
    Encodes neccesary information for compressed image into a binary .lrk (low rank) custom file

    Args:
        metadata (6 tuple): Tuple containing neccesary information for image reconstruction

    Returns:
        None
"""
def write_output(metadata: tuple, image_name: str) -> None:

    out_path = f"../bin/{image_name}_compressed.rnk"

    U_k, Sigma_k, VT_k, Height, Width, A_kDataType = metadata

    with open(out_path, 'wb') as f:
        # Turn all the data into utf-8





    



'''
Complete writing to output file and finish all encodings
Write decoder
Update github with lots of info


'''