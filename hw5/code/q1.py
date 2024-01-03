# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# Nov, 2023
###################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
from utils import plotSurface
from utils import integrateFrankot


def renderNDotLSphere(center, rad, light, pxSize, res):
    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0] / 2) * pxSize * 1.0e-4
    Y = (Y - res[1] / 2) * pxSize * 1.0e-4
    Z = np.sqrt(rad**2 + 0j - X**2 - Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)

    image = None
    # Your code here
    image = np.zeros((res[1], res[0]))
    for i in range(res[1]):
        for j in range(res[0]):
            normal = np.array([X[i, j], Y[i, j], Z[i, j]])
            normal /= (np.linalg.norm(normal) + 1.0e-10)
            image[i, j] = np.dot(normal, light)
    image[image < 0] = 0
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def loadData(path="../data/"):
    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None
    # Your code here
    for i in range(7):
        image = plt.imread(path + "input_" + str(i + 1) + ".tif")

        # make sure datatype is uint16
        image = image.astype(np.uint16)

        # convert to xyz
        image = rgb2xyz(image)

        # extract luminance
        luminance = image[:, :, 1]
        
        if I is None:
            I = np.zeros((7, luminance.size))
            s = luminance.shape
        
        # stack luminance
        I[i, :] = luminance.reshape((1, luminance.size))

    # load sources and convert to 3x7
    L = np.load(path + "sources.npy").T
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):
    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    # Your code here
    B = np.linalg.inv(L @ L.T) @ L @ I
    return B


def estimateAlbedosNormals(B):
    """
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    """

    albedos = None
    normals = None
    # Your code here
    albedos = np.linalg.norm(B, axis=0)
    normals = B / (albedos + 1.0e-10)
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None
    # Your code here
    albedoIm = albedos.reshape(s)
    normalIm = normals.T.reshape((s[0], s[1], 3))

    # normalize albedo
    albedoIm = (albedoIm - np.min(albedoIm)) / (np.max(albedoIm) - np.min(albedoIm))

    # normalize normals
    normalIm = (normalIm - np.min(normalIm)) / (np.max(normalIm) - np.min(normalIm))
    return albedoIm, normalIm


def estimateShape(normals, s):
    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    # Your code here
    surface = integrateFrankot((-normals[0, :]/(normals[2, :] + 1.0e-10)).reshape(s), (-normals[1, :]/(normals[2, :] + 1.0e-10)).reshape(s), s)
    return surface


if __name__ == "__main__":
    # Part 1(b)
    radius = 0.75  # cm
    center = np.asarray([0, 0, 0])  # cm
    pxSize = 7  # um
    res = (3840, 2160)

    # light = np.asarray([1, 1, 1]) / np.sqrt(3)
    # image = renderNDotLSphere(center, radius, light, pxSize, res)
    # plt.figure()
    # plt.imshow(image, cmap="gray")
    # plt.imsave("1b-a.png", image, cmap="gray")

    # light = np.asarray([1, -1, 1]) / np.sqrt(3)
    # image = renderNDotLSphere(center, radius, light, pxSize, res)
    # plt.figure()
    # plt.imshow(image, cmap="gray")
    # plt.imsave("1b-b.png", image, cmap="gray")

    # light = np.asarray([-1, -1, 1]) / np.sqrt(3)
    # image = renderNDotLSphere(center, radius, light, pxSize, res)
    # plt.figure()
    # plt.imshow(image, cmap="gray")
    # plt.imsave("1b-c.png", image, cmap="gray")

    # Part 1(c)
    I, L, s = loadData("./hw5/data/")

    # Part 1(d)
    # Your code here
    U, S, V = np.linalg.svd(I, full_matrices=False)
    print(S)

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)

    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    # plt.imsave("./hw5/results/1f-a.png", albedoIm, cmap="gray")
    # plt.imsave("./hw5/results/1f-b.png", normalIm, cmap="rainbow")

    # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface)
