import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import random
import scipy.optimize as opt
from q3_2_triangulate import triangulate
# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()

"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""
def ransacF(noisy_pts1, noisy_pts2, M, nIters=100, tol=10):
    best_inliers = None
    bestF = None
    maxCount = 0

    homogenous_pts1 = toHomogenous(noisy_pts1)
    homogenous_pts2 = toHomogenous(noisy_pts2)

    for i in range(int(nIters)):
        randomIndices = np.array(random.sample(range(len(noisy_pts1)), 10))

        selected_pts1 = noisy_pts1[randomIndices]
        selected_pts2 = noisy_pts2[randomIndices]
        F = eightpoint(selected_pts1, selected_pts2, M)

        epi_error = calc_epi_error(homogenous_pts1, homogenous_pts2, F)
        
        inliers = epi_error < tol
        inliers_count = np.count_nonzero(inliers)

        if inliers_count > maxCount:
            bestF = F
            maxCount = inliers_count
            best_inliers = inliers
    
    print('Inlier Count: {:.2f}'.format(maxCount))
    return bestF, best_inliers

"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""
def rodrigues(r):
    theta = np.linalg.norm(r)
    
    u = r/theta
    u_x = np.array([[0, -u[2], u[1]],
                    [u[2], 0, -u[0]], 
                    [-u[1], u[0], 0]])
    u = u.reshape(-1, 1)
    return np.eye(3)*np.cos(theta) + np.sin(theta)*u_x  + (1-np.cos(theta))*(u@u.T) 

"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""
def invRodrigues(R):
    A = (R - R.T)/2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]]).T
    s = np.linalg.norm(rho)
    c = (np.trace(R) - 1)/2
    theta = np.arctan2(s, c)

    def S(r):
        if np.linalg.norm(r) == np.pi and ((r[0]==0 and r[1]==0 and r[2]<0) or (r[0]==0 and r[1]<0) or (r[0]<0)):
            return -r
        else:
            return r

    if s == 0 and c == 1:
        return np.zeros(3)
    elif s == 0 and c == -1:
        v = np.diag(R) + 1
        u = v/np.linalg.norm(v)
        r = S(u*np.pi)
    elif np.sin(theta) != 0:
        u = rho/s
        r = u*theta

    return r

"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    r2 = x[3*len(p1):3*len(p1)+3]
    t2 = x[3*len(p1)+3:]
    x = x[:3*len(p1)].reshape(len(p1), 3)

    P = np.hstack((x, np.ones((len(p1),1))))
    M2 = np.hstack((rodrigues(r2), t2.reshape(-1, 1)))

    C1 = K1 @ M1
    C2 = K2 @ M2

    p1_hat = C1 @ P.T
    p1_hat = p1_hat/p1_hat[-1]
    p2_hat = C2 @ P.T
    p2_hat = p2_hat/p2_hat[-1]

    residuals = np.concatenate([(p1 - p1_hat[:2].T).reshape([-1]), (p2 - p2_hat[:2].T).reshape([-1])])
    return residuals

"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    r2 = invRodrigues(M2_init[:, :3])
    t2 = M2_init[:, 3]
    obj_start = np.concatenate((P_init.flatten(), r2.flatten(), t2))

    print('Initial reprojection error: ', np.sum(rodriguesResidual(K1, M1, p1, K2, p2, obj_start)**2))
    obj_end = opt.minimize(lambda x: np.sum(rodriguesResidual(K1, M1, p1, K2, p2, x)**2), obj_start).x
    print('Optimised reprojection error: ', np.sum(rodriguesResidual(K1, M1, p1, K2, p2, obj_end)**2))
    
    P = obj_end[:3*len(p1)].reshape(len(p1), 3)
    r2 = obj_end[3*len(p1):3*len(p1)+3]
    t2 = obj_end[3*len(p1)+3:]
    
    M2 = np.concatenate((rodrigues(r2), t2.reshape(-1, 1)), axis=1)
    return M2, P, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    eightpointF = eightpoint(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=100, tol=10)

    # displayEpipolarF(im1, im2, F)
    selected_pts1 = noisy_pts1[inliers]
    selected_pts2 = noisy_pts2[inliers]
    F, _ = ransacF(selected_pts1, selected_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=100, tol=10)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(
        noisy_pts2
    )

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    M2, C2, Pb = findM2(F, selected_pts1, selected_pts2, intrinsics)
    M2, Pa, obj_start, obj_end = bundleAdjustment(K1, M1, selected_pts1, K2, M2, selected_pts2, Pb)

    print("Eight Point F: ", eightpointF) 
    print("RANSAC F: ", F)
    print("Refined F: ", F)
    # plot_3D_dual(Pb, Pa)