import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    ################### TODO Implement Inverse Composition Affine ###################
    p = np.zeros(6)

    spline = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    spline1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    X,Y = np.meshgrid(np.linspace(0, It1.shape[1]-1, It1.shape[1], endpoint=True), np.linspace(0, It1.shape[0]-1, It1.shape[0], endpoint=True))
    Y, X = Y.flatten(), X.flatten()
    coords = np.vstack((X, Y, np.ones(X.size)))

    # Warp gradient 
    dItx = spline.ev(Y, X, dx=1).flatten()
    dIty = spline.ev(Y, X, dy=1).flatten()

    # Evaluate Jacobian
    # jacobian = np.array([[X,0,Y,0,1,0], [0,X,0,Y,0,1]])

    # Compute approximate Hessian
    # preH = dIt @ jacobian Crashes
    preH = np.array([dIty*X, dIty*Y, dIty, dItx*X, dItx*Y, dItx]).T
    H = preH.T @ preH
    postH = np.linalg.inv(H) @ preH.T

    for i in range(int(num_iters)):
        # Warp I with W(x;p) to compute I(W(x;p))
        # WIt1 = affine_transform(coords, M) Slow
        WIt1 = M @ coords

        # Pixels common to It and WIt1
        outside = np.nonzero(((WIt1[0] < 0) | (WIt1[0] >= It.shape[1])) | (WIt1[1] < 0) | (WIt1[1] >= It.shape[0]))

        # Compute error image T(x) - I(W(x;p))
        error = spline.ev(Y, X) - spline1.ev(WIt1[1], WIt1[0])
        error[outside] = 0

        # Debug
        # import pdb; pdb.set_trace()

        # Compute dp
        dp = postH @ error.reshape(-1, 1)

        # Update the warp parameters
        p = p + dp.flatten()

        # Update M
        delM = np.array([[1.0+p[0], p[1], p[2]], [p[3], 1.0+p[4], p[5]], [0.0, 0.0, 1.0]], dtype=np.float)
        M = M @ np.linalg.inv(delM)

        # Check for convergence
        if np.linalg.norm(dp)**2 < threshold:
            break

    return M
