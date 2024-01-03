import numpy as np
import matplotlib.pyplot as plt
import os
from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate
"""
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.

Modified by Vineet Tambe, 2023.
"""
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=100):
    for i in range(pts1.shape[0]):
        confidence1 = pts1[i, 2]
        confidence2 = pts2[i, 2]
        confidence3 = pts3[i, 2]

        if(confidence1 < confidence2 and confidence1 < confidence3 and confidence1 > Thres):
            p, e = triangulate(C2, np.array([pts2[i, :2]]), C3, np.array([pts3[i, :2]]))
        elif(confidence2 < confidence1 and confidence2 < confidence3 and confidence2 > Thres):
            p, e = triangulate(C3, np.array([pts3[i, :2]]), C1, np.array([pts1[i, :2]]))
        elif(confidence3 > Thres):
            p, e = triangulate(C1, np.array([pts1[i, :2]]), C2, np.array([pts2[i, :2]]))
        else:
            continue

        if i == 0:
            P = p
            err = e
        else:
            P = np.vstack((P, p))
            err = np.hstack((err, e))
    return P, err

"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""
def plot_3d_keypoint_video(pts_3d_video):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    for i in range(len(pts_3d_video)):
        pts_3d = pts_3d_video[i]
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d[index0, 0], pts_3d[index1, 0]]
            yline = [pts_3d[index0, 1], pts_3d[index1, 1]]
            zline = [pts_3d[index0, 2], pts_3d[index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j])
        np.set_printoptions(threshold=1e6, suppress=True)

    plt.show()

# Extra Credit
if __name__ == "__main__":
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join("data/q6/", "time" + str(loop) + ".npz")
        image1_path = os.path.join("data/q6/", "cam1_time" + str(loop) + ".jpg")
        image2_path = os.path.join("data/q6/", "cam2_time" + str(loop) + ".jpg")
        image3_path = os.path.join("data/q6/", "cam3_time" + str(loop) + ".jpg")

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data["pts1"]
        pts2 = data["pts2"]
        pts3 = data["pts3"]

        K1 = data["K1"]
        K2 = data["K2"]
        K3 = data["K3"]

        M1 = data["M1"]
        M2 = data["M2"]
        M3 = data["M3"]

        # Note - Press 'Escape' key to exit img preview and loop further
        # img = visualize_keypoints(im2, pts2)
        # TODO: YOUR CODE HERE
        C1 = K1 @ M1
        C2 = K2 @ M2
        C3 = K3 @ M3

        P, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)
        pts_3d_video.append(P)
        # print("Error: ", np.mean(err))
        np.savez("submission/q6_2.npz", P=P)

        plot_3d_keypoint_video(pts_3d_video)

        
