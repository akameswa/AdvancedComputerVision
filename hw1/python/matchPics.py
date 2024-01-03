import numpy as np
import cv2
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts):
        
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

        # TODO: Convert Images to GrayScale
        I1G = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I2G = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        
        # TODO: Detect Features in Both Images
        fea1 = corner_detection(I1G, sigma)
        fea2 = corner_detection(I2G, sigma)
        
        # TODO: Obtain descriptors for the computed feature locations
        desc1, locs1 = computeBrief(I1G, fea1)
        desc2, locs2 = computeBrief(I2G, fea2)

        # TODO: Match features using the descriptors
        matches = briefMatch(desc1, desc2, ratio)

        # Fliping y and x to x and y
        locs1 = np.fliplr(locs1)
        locs2 = np.fliplr(locs2)

        return matches, locs1, locs2
