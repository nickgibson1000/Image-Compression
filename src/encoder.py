import struct
import numpy as np

"""
    Encodes neccesary information for compressed image into a binary .lrk (low rank) custom file

    Args:
        metadata (6 tuple): Tuple containing neccesary information for image reconstruction

    Returns:
        None
"""
def write_output(image_name: str, metadata: tuple) -> None:

    out_path = f"../bin/{image_name}_compressed.rnk"

    U_k, Sigma_k, VT_k, Height, Width, A_kDataType = metadata

    with open(out_path, 'wb') as f:
        # Turn all the data into utf-8?

        write_matrix(U_k, f)
        write_matrix(Sigma_k, f)
        write_matrix(VT_k, f)



def write_matrix(matrix, file_handle):
    
    flat = matrix.flatten()
    
    # Header
    file_handle.write(struct.pack('<ii', matrix.shape[0], matrix.shape[1]))
    
    # Data
    file_handle.write(struct.pack('<' + 'f'*len(flat), *flat))

