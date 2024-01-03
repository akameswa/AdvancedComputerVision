import numpy as np
import cv2
import matplotlib.pyplot as plt
from opts import get_opts
from helper import loadVid
from matchPics import matchPics
from planarH import compositeH
from planarH import computeH_ransac
opts = get_opts()

# Load videos and cover image
ar = loadVid('C:/D/CMU/Courses/F23/16-820/hw1/data/ar_source.mov')
book = loadVid('C:/D/CMU/Courses/F23/16-820/hw1/data/book.mov')
cv_cover = cv2.imread('C:/D/CMU/Courses/F23/16-820/hw1/data/cv_cover.jpg')

output_video = cv2.VideoWriter('C:/D/CMU/Courses/F23/16-820/hw1/data/ar_result.avi', cv2.VideoWriter_fourcc('F','M','P','4'), 30, (book.shape[2], book.shape[1]))

for i in range(ar.shape[0]):
    print('Frame:', i)

    # Extracting current frames
    ar_image = ar[i]
    book_image = book[i]

    # Removing black padding
    ar_image_without_black_padding = ar_image[45:310,:,:]
    
    # Resizing AR to match cover
    ratio = cv_cover.shape[0]/ar_image_without_black_padding.shape[0]
    resize_width = int(ar_image_without_black_padding.shape[1]*ratio)
    resize_height = int(ar_image_without_black_padding.shape[0]*ratio)
    resized_ar = cv2.resize(ar_image_without_black_padding, (resize_width, resize_height))
    
    # Cropping AR
    ar_crop = ar_image_without_black_padding[:,(int(resize_width/2)-int(cv_cover.shape[1]/2)):(int(resize_width/2)+int(cv_cover.shape[1]/2)),:]
    resized_ar = cv2.resize(ar_crop, (cv_cover.shape[1], cv_cover.shape[0]))

    matches, locs1, locs2 = matchPics(cv_cover,book_image,opts)

    # Skip frame if not enough matches
    if len(matches) < 5:
        print('Skipping frame', i)
        continue
    
    # Extracting locations of matches
    locs1 = locs1[matches[:, 0], :]
    locs2 = locs2[matches[:, 1], :]

    H,_ = computeH_ransac(locs1, locs2, opts)

    output = compositeH(H, resized_ar, book_image)
    
    output_video.write(output)

output_video.release()