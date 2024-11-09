# 16-820 Advanced Computer Vision Projects

A comprehensive implementation of advanced computer vision algorithms including image matching, object tracking, 3D reconstruction, neural networks, and surface reconstruction. Features practical applications like AR overlays, panorama creation, motion detection, and character recognition developed as part of CMU's 16-820 course.

## Project Structure

- `hw1/`: Image matching, homography, and AR applications
- `hw2/`: Lucas-Kanade tracking and motion detection
- `hw3/`: Epipolar geometry and 3D reconstruction
- `hw4/`: Neural networks and object detection
- `hw5/`: Photometric stereo and surface reconstruction

## Key Features

### Image Processing & AR
- FAST corner detection and BRIEF feature matching
- Homography computation using RANSAC
- AR overlay on book covers and panorama creation

### Object Tracking
- Lucas-Kanade tracking algorithm
- Template correction for drift prevention
- Motion detection and segmentation

### 3D Reconstruction
- Eight-point algorithm for fundamental matrix estimation
- Triangulation for 3D point reconstruction
- Bundle adjustment optimization

### Neural Networks & OCR
- CNN implementation for character recognition
- ResNet50 for object detection
- Text detection and line segmentation

### Surface Reconstruction
- Photometric stereo implementation
- Surface normal estimation
- Depth map integration

## Dependencies
- NumPy
- OpenCV
- PyTorch
- Scikit-image
- Matplotlib

