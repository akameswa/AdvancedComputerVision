import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    # Estimate noise
    noise = skimage.restoration.estimate_sigma(image, average_sigmas=True)
    win_size = max(5, 2*np.ceil(3*noise)+1)
    # Denoise
    denoisedImage = skimage.restoration.denoise_bilateral(image, win_size=win_size, channel_axis=-1)
    # Greyscale
    greyscaleImage = skimage.color.rgb2gray(denoisedImage)
    # Threshold
    threshold = skimage.filters.threshold_otsu(greyscaleImage)

    # Morphology
    bw = skimage.morphology.closing(greyscaleImage<threshold, skimage.morphology.square(10))
    # Label
    label = skimage.measure.label(bw)
    # Skip small boxes
    regions = skimage.measure.regionprops(label)
    
    for region in regions:
        if region.area > 100:
            bboxes.append(region.bbox)

    return bboxes, bw
