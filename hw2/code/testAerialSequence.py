import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=5,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.2,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq',
    default='C:/D/CMU/Courses/F23/16-820/hw2/data/aerialseq.npy',
)

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
seq_file_path = args.seq

seq = np.load(seq_file_path)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''
for i in range(seq.shape[2]-1):
    mask = SubtractDominantMotion(seq[:,:,i], seq[:,:,i+1], threshold, num_iters, tolerance)

    if i in [30, 60, 90, 120]:
        plt.figure(figsize=(10, 10))
        plt.imshow(seq[:,:,i], cmap='gray')
        plt.imshow(mask, alpha=0.45, cmap='Blues')
        plt.axis('off')
        plt.savefig('C:/D/CMU/Courses/F23/16-820/hw2/submission/results/aerial_{}'.format(i))