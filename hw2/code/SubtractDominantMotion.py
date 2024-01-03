import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.zeros(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominent Motion ###################
    # 1. Warp the image It using M
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    WIt = affine_transform(image1, M)

    # 2. Subtract It from It+1
    difference = abs(image2 - WIt)

    # 3. Motion where the absolute difference exceeds a threshold
    mask[difference < tolerance] = 0  
    mask[difference > tolerance] = 1

    # 4. Erode and dilate the mask
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    mask = binary_dilation(mask, struct, iterations=5)
    mask = binary_erosion(mask, struct, iterations=5)

    return mask.astype(bool)
