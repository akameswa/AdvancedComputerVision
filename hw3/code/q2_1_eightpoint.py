import numpy as np
import matplotlib.pyplot as plt
from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize
"""
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
"""
def eightpoint(pts1, pts2, M):
    npts1 = pts1/M
    npts2 = pts2/M

    A = np.zeros((npts1.shape[0], 9))
    A[:, 0] = npts1[:, 0] * npts2[:, 0]
    A[:, 1] = npts1[:, 1] * npts2[:, 0]
    A[:, 2] = npts2[:, 0]
    A[:, 3] = npts1[:, 0] * npts2[:, 1]
    A[:, 4] = npts1[:, 1] * npts2[:, 1]
    A[:, 5] = npts2[:, 1]
    A[:, 6] = npts1[:, 0]
    A[:, 7] = npts1[:, 1]
    A[:, 8] = 1

    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    F = refineF(F, npts1, npts2)
    F = F / F[2, 2]
    
    scaleT = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    F = scaleT.T @ F @ scaleT
    return F


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    
    M = np.max([*im1.shape, *im2.shape])
    F = eightpoint(pts1, pts2, M)
    print("\n F = ", F)
    print("\n M", M)
    np.savez("submission/q2_1.npz", F=F, M=M)

    # Q2.1
    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
