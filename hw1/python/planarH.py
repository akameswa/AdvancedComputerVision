import numpy as np
import cv2
import random

def computeH(x1, x2):
    #Q2.2.1
    # TODO: Compute the homography between two sets of points
    A = []
    
    # Extract the x and y coordinates from the first set of points
    u1 = x1[:,0]
    v1 = x1[:,1]

    # Extract the x and y coordinates from the second set of points
    u2 = x2[:,0]
    v2 = x2[:,1]

    # Get the number of points
    index = x1.shape[0]

    for i in range(index):
        # Construct matrix A based on the homography equation
        A.append([u2[i], v2[i], 1, 0, 0, 0, -u1[i]*u2[i] , -u1[i]*v2[i], -u1[i]])
        A.append([0, 0, 0, u2[i], v2[i], 1, -v1[i]*u2[i], -v1[i]*v2[i], -v1[i]])

    # Convert list to array
    A = np.vstack(A)

    # Perform SVD
    U, S, V = np.linalg.svd(A)

    # Last column of V gives the solution
    eigen_vector = V.T[:,-1]

    H2to1 = eigen_vector.reshape(3,3)

    return H2to1


def computeH_norm(x1, x2):

    #Q2.2.2
    # TODO: Compute the centroid of the points
    u1 = np.array(x1[:,0])
    v1 = np.array(x1[:,1])

    u2 = np.array(x2[:,0])
    v2 = np.array(x2[:,1])

    u1_mean = np.mean(u1)
    v1_mean = np.mean(v1)

    u2_mean = np.mean(u2)
    v2_mean = np.mean(v2)

    # TODO: Shift the origin of the points to the centroid
    u1_hat = u1 - u1_mean
    v1_hat = v1 - v1_mean

    u2_hat = u2 - u2_mean
    v2_hat = v2 - v2_mean

    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_dist = np.sqrt(np.square(u1_hat)+np.square(v1_hat))
    x2_dist = np.sqrt(np.square(u2_hat)+np.square(v2_hat))

    scale1 = np.sqrt(2)/np.max(x1_dist)
    scale2 = np.sqrt(2)/np.max(x2_dist)

    x1_scaled = np.array([u1_hat, v1_hat])*scale1
    x2_scaled = np.array([u2_hat, v2_hat])*scale2

    # TODO: Similarity transform 1 and 2
    S1 = np.array(([scale1, 0, 0],
                [0, scale1, 0],
                [0, 0, 1]))
    S2 = np.array(([scale2, 0, 0],
                [0, scale2, 0],
                [0, 0, 1]))

    L1 = np.array(([1, 0, -u1_mean],
                [0, 1, -v1_mean],
                [0, 0, 1]))
    L2 = np.array(([1, 0, -u2_mean],
                [0, 1, -v2_mean],
                [0, 0, 1]))
    T1 = S1@L1
    T2 = S2@L2

    # TODO: Compute homography
    H = computeH(x1_scaled.T, x2_scaled.T)
    
    # TODO: Denormalization
    H2to1 = np.linalg.inv(T1)@H@T2
    return H2to1


def computeH_ransac(locs1, locs2, opts):
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    # Initializing variables
    bestCount = 0
    inlierScore = np.zeros(locs1.shape[0], dtype=int)

    for i in range(max_iters):
        # Generating 4 random points
        randomPoints = np.array(random.sample(range(locs1.shape[0]),4))
    
        # Selecting 4 points from locs1 and locs2
        randomLocs1 = locs1[randomPoints]
        randomLocs2 = locs2[randomPoints]

        H = computeH_norm(randomLocs1,randomLocs2)

        # Homogenising
        locs1_hom = np.hstack((locs1, np.ones((locs1.shape[0], 1))))
        locs2_hom = np.hstack((locs2, np.ones((locs2.shape[0], 1))))

        # Estimating locs2
        locs1_estimated = H@locs2_hom.T
        locs1_estimated_norm = (locs1_estimated / locs1_estimated[2, :]).T

        # Computing error between original and estimated points
        inliers = locs1_hom - locs1_estimated_norm
        error = np.linalg.norm(inliers, axis = 1)

        # Counting inliers
        score = np.where(error < inlier_tol, 1, 0)
        count = np.sum(score)

        if count > bestCount:
            inlierScore = score
            bestCount = count

    # Computing best homography
    inliers_list = np.array([locs1[inlierScore==1], locs2[inlierScore==1]])
    bestH = computeH_norm(inliers_list[0], inliers_list[1])

    return bestH, score



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    
    # Inverting Homography for warping
    H2to1 = np.linalg.inv(H2to1)

    # TODO: Create mask of same size as template
    mask = np.ones_like(template)

    # TODO: Warp mask by appropriate homography
    warpedMask = cv2.warpPerspective(mask,H2to1,(img.shape[1], img.shape[0]))

    # TODO: Warp template by appropriate homography
    warpedTemplate = cv2.warpPerspective(template,H2to1,(img.shape[1], img.shape[0]))

    # TODO: Use mask to combine the warped template and the image
    img[np.nonzero(warpedMask)] = warpedTemplate[np.nonzero(warpedMask)]
    
    return img




