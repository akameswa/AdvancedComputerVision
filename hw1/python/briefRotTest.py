import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
import matplotlib.pyplot as plt
from helper import plotMatches

#Q2.1.6

def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    I1G = cv2.imread('data/cv_cover.jpg')
    # I1G = cv2.cvtColor(I1G, cv2.COLOR_BGR2GRAY)

    x_axis_values = []
    y_axis_values = []

    for i in range(36):

        # TODO: Rotate Image
        R1 = scipy.ndimage.rotate(I1G, 10*i, axes=(1,0))

        # TODO: Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(I1G, R1, opts)
        # plotMatches(I1G, R1, matches, locs1, locs2)
    
        # TODO: Update histogram
        x_axis_values.append(i)
        y_axis_values.append(matches.shape[0])

    pass 

    # TODO: Display histogram
    plt.xlabel('Rotation')
    plt.ylabel('Number of matches')
    plt.bar(x_axis_values,y_axis_values)

if __name__ == "__main__":
    
    opts = get_opts()
    rotTest(opts)
