import numpy as np
import cv2
from opts import get_opts
import matplotlib.pyplot as plt
from matchPics import matchPics
from planarH import compositeH
from planarH import computeH_ransac
opts = get_opts()

# Q4
left = cv2.imread('C:/D/CMU/Courses/F23/16-820/hw1/data/lft.jpeg')
right = cv2.imread('C:/D/CMU/Courses/F23/16-820/hw1/data/rght.jpeg')

# Add padding
rightPadded = cv2.copyMakeBorder(right, 0, 0, right.shape[1]//5, 0, cv2.BORDER_CONSTANT)

matches, locs1, locs2 = matchPics(left,rightPadded,opts)

locs1 = locs1[matches[:, 0], :]
locs2 = locs2[matches[:, 1], :]

H,_ = computeH_ransac(locs1, locs2, opts)

output = compositeH(H, left, rightPadded)

cv2.imwrite('C:/D/CMU/Courses/F23/16-820/hw1/data/test.jpg',output)