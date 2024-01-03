import numpy as np
import cv2
from helper import loadVid
from planarH import compositeH
import time

def matchPics(I1, I2):
    orb = cv2.ORB_create()
    
    locs1, desc1 = orb.detectAndCompute(I1, mask=None)
    locs2, desc2 = orb.detectAndCompute(I2, mask=None)

    # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    # Given: 1. For ORB, use NORM_HAMMING 2.Keep crossCheck=True for best match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1,desc2)

    # Extracting locations of matches after converting DMatch objects
    points1 = np.array([cv2.KeyPoint_convert([locs1[match.queryIdx]]) for match in matches])
    points2 = np.array([cv2.KeyPoint_convert([locs2[match.trainIdx]]) for match in matches])

    return points1, points2

def main():
    # Load videos and cover image
    ars = loadVid('C:/D/CMU/Courses/F23/16-820/hw1/data/ar_source.mov')
    books = loadVid('C:/D/CMU/Courses/F23/16-820/hw1/data/book.mov')
    cv_cover = cv2.imread('C:/D/CMU/Courses/F23/16-820/hw1/data/cv_cover.jpg')

    # Removing black padding
    ar = ars[:,45:310,:,:]

    # Resizing AR to match cover
    ratio = cv_cover.shape[0]/ar[0].shape[0]
    resize_width = int(ar[0].shape[1]*ratio)

    initial_time = time.time()
    previous_time = 0

    for i in range(ar.shape[0]):
        book = books[i]

        ar_crop = ar[i][:,(int(resize_width/2)-int(cv_cover.shape[1]/2)):(int(resize_width/2)+int(cv_cover.shape[1]/2)),:]
        resized_ar = cv2.resize(ar_crop, dsize=(cv_cover.shape[1],cv_cover.shape[0]))
        
        locs1, locs2 = matchPics(cv_cover,book)

        H,_ = cv2.findHomography(locs2, locs1, cv2.RANSAC, 6.0)
        
        composite_img = compositeH(H, resized_ar, book)

        cv2.imshow('composite_img', composite_img)
        cv2.waitKey(1)

        current_time = time.time() - initial_time
        time_difference = current_time - previous_time
        frame_rate = 1 / time_difference
        print('Frame rate:', frame_rate)
        previous_time = current_time

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    