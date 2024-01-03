import numpy as np
import matplotlib.pyplot as plt
from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF
"""
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
"""
def sevenpoint(pts1, pts2, M):
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
    F1 = V[-1].reshape(3, 3)
    F2 = V[-2].reshape(3, 3)
    Fmat = [F1, F2]
   
    D = np.zeros((2, 2, 2))
    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):
                Dtmp = np.zeros((3, 3))
                Dtmp[:, 0] = Fmat[i1][:, 0]
                Dtmp[:, 1] = Fmat[i2][:, 1]
                Dtmp[:, 2] = Fmat[i3][:, 2]
                D[i1, i2, i3] = np.linalg.det(Dtmp)

    coefficients = np.zeros(4)
    coefficients[0] = D[1, 1, 1]
    coefficients[1] = D[1, 1, 0] + D[0, 1, 1] + D[1, 0, 1] - 3 * D[1, 1, 1]
    coefficients[2] = D[0, 0, 1] - 2 * D[0, 1, 1] - 2 * D[1, 0, 1] + D[1, 0, 0] - 2 * D[1, 1, 0] + D[0, 1, 0] + 3 * D[1, 1, 1]
    coefficients[3] = -D[1, 0, 0] + D[0, 1, 1] + D[0, 0, 0] + D[1, 1, 0] + D[1, 0, 1] - D[0, 1, 0] - D[0, 0, 1] - D[1, 1, 1]

    solutions = np.roots(coefficients)
    scaleT = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    
    Farray = [root.real * F1 + (1 - root.real) * F2 for root in solutions if root.imag == 0]

    for i in range(len(Farray)):
        Farray[i] = refineF(Farray[i], npts1, npts2)
        Farray[i] /= Farray[i][2, 2]
        Farray[i] = scaleT.T @ Farray[i] @ scaleT
        
    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[0]

    np.savez("submission/q2_2.npz", F, M)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 100
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])
    print('Recovered F:', F)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
