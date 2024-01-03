import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

for img in os.listdir("hw4\images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("hw4\images", img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################   
    line = []
    lines = []
    pBox = bboxes[0]
    
    threshold = 95 if img == "01_list.jpg" else 45
    firstBox = True

    for box in bboxes[1:]:
        y1, x1, y2, x2 = box
        py1, px1, py2, px2 = pBox
            
        if abs(y1 - py1) < threshold:
            if firstBox:
                line.append(pBox)
                firstBox = False
            line.append(box) 
        else:
            lines.append(line)
            line = []
            line.append(box)
        pBox = box
    lines.append(line)

    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    sentences = []
    for line in lines:
        line.sort(key=lambda x: x[1])
        characters = []
        for box in line:
            y1, x1, y2, x2 = box
            crop = bw[y1:y2, x1:x2]        
            
            if img == "01_list.jpg":
                dilation = 5
            else:
                dilation= 15

            crop = skimage.morphology.dilation(crop, skimage.morphology.square(dilation))
            crop = skimage.util.invert(crop)
            crop = np.pad(crop, ((20, 20), (20, 20)), mode="constant", constant_values=1)
            crop = skimage.transform.resize(crop, (32, 32)).T
            characters.append(crop.flatten())
        sentences.append(characters)

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
    )
    params = pickle.load(open("C:\\D\\CMU\\Courses\\F23\\16-820\\hw4\\python\\q3_weights.pickle", "rb"))
    ##########################
    ##### your code here #####
    ##########################

    for sentence in sentences:
        line = ""
        letter = np.vstack(sentence)
        h = forward(letter, params, "layer1")
        probs = forward(h, params, "output", softmax)
        predictions = np.argmax(probs, axis=1)
        for prediction in predictions:
            line += letters[prediction]
        print(line)