# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# 
# Nov, 2023
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import (
    loadData,
    estimateAlbedosNormals,
    displayAlbedosNormals,
    estimateShape,
)
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface

def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    B = None
    L = None
    # Your code here
    U, S, V = np.linalg.svd(I, full_matrices=False)
    B = V[:3, :]
    L = U[:3, :]
    return B, L


def plotBasRelief(B, mu, nu, lam):
    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter

    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    # Your code here
    G = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    _, normals = estimateAlbedosNormals(np.linalg.inv(G.T) @ B)
    normals = enforceIntegrability(normals, s)
    surface = estimateShape(normals, s)
    plotSurface(surface)

if __name__ == "__main__":
    # Part 2 (b)
    # Your code here
    I, L0, s = loadData("./hw5/data/")
    B, L = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # plt.imsave("./hw5/results/2b-a.png", albedoIm, cmap="gray")
    # plt.imsave("./hw5/results/2b-b.png", normalIm, cmap="rainbow")

    # Part 2 (d)
    # Your code here
    surface = estimateShape(normals, s)
    # plotSurface(surface)

    # Part 2 (e)
    # Your code here
    normals = enforceIntegrability(normals, s)
    G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    normals = G @ normals
    surface = estimateShape(normals, s)
    # plotSurface(surface)

    # Part 2 (f)
    # Your code here
    plotBasRelief(B, 0, 0, 10)
