import numpy as np


"""
    Calculates the Singular Values Decomposition(SVD) of an image

    Args:
        image_matrix_greyscale (?): ?

    Returns:
        ?(tuple?): The completed singular value decomposition matricies
"""
#TO-DO: Make sure this is correct
def svd(image_matrix_greyscale):

    A = image_matrix_greyscale.astype(np.float64)
    AT = A.T

    new_matrix = AT @ A

    # To-Do: Create own function finding eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(new_matrix)
    

    # Sort sigular values in descending order and 
    # Convert all floating point error eigenvalues < 0 to 0
    singular_values = np.sqrt(np.clip(eigenvalues, 0, None))
    singular_values = np.sort(singular_values)[::-1]

    print(f"Number of singular values: {len(singular_values)}")

    # Sigma
    S = np.zeros((len(singular_values), len(singular_values)))

    # Fill the diagonals of S with the singular values
    # so that the sigmas go in descending value
    # np.fill_diagonal(S, singular_values) -> Simpler
    for i in range(len(S)):
        S[i][i] = singular_values[i]


    #V = np.zeroes((len(eigenvectors), len(eigenvectors)))

    #V = np.zeros((len(singular_values), len(singular_values)))
    V = np.fliplr(eigenvectors)

#----------------------------------------------------------------------------------------------------------------------------------------------
    # Now we will find our U matrix
    U = np.zeros_like(A, dtype=np.float64)

    for i, sigma in enumerate(singular_values):
        if sigma > 1e-12:
            vi = V[:, i]
            U[:, i] = (A @ vi) / sigma
        else:
            # leave it for now — fill later
            pass

    svd = (U, S, V.T) 
    #test_matricies(svd)

    return svd


def calc_eigenvalues():


    print()