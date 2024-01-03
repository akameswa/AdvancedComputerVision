import numpy as np
import cv2
import skimage.color
from opts import get_opts
import matplotlib.pyplot as plt

from matchPics import matchPics
from planarH import compositeH
from planarH import computeH_ransac


def warpImage(opts):
    I1 = cv2.imread('data\cv_cover.jpg')
    I2 = cv2.imread('data\cv_desk.png')

    I3 = cv2.imread('data\hp_cover.jpg')
    I3 = cv2.resize(I3, (I1.shape[1], I1.shape[0]))

    matches, locs1, locs2 = matchPics(I1,I2,opts)

    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]

    H,_ = computeH_ransac(locs1, locs2, opts)
    
    output = compositeH(H, I3, I2)

    cv2.imshow('Image',output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


