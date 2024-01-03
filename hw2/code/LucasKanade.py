import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    p = p0
    x1, y1, x2, y2 = rect

    spline = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    spline1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    patchY, patchX = np.meshgrid(np.linspace(y1, y2, int(y2-y1+1), endpoint=True), np.linspace(x1, x2, int(x2-x1+1), endpoint=True))

    for i in range(int(num_iters)):
        # Warp I with W(x;p) to compute I(W(x;p))
        WIt1 = spline1.ev(patchY + p[1], patchX + p[0])

        # Compute error image T(x) - I(W(x;p))
        error = spline.ev(patchY, patchX) - WIt1

        # Warp gradient 
        dItx = spline1.ev(patchY + p[1], patchX + p[0], dx=1).reshape(1,-1)
        dIty = spline1.ev(patchY + p[1], patchX + p[0], dy=1).reshape(1,-1)
        dIt = np.hstack((dIty.T, dItx.T))

        # Evaluate Jacobian
        jacobian = np.array([[1, 0], [0, 1]])

        # Compute approximate Hessian
        preH = dIt @ jacobian
        H = preH.T @ preH
        
        # Compute dp
        dp = np.linalg.inv(H) @ preH.T @ error.reshape(-1, 1) 
        
        # Update the warp parameters
        p = p + dp.flatten()

        # Check for convergence
        if np.linalg.norm(dp)**2 < threshold:
            break

    return p