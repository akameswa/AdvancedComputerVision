import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=0.01,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.03,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq_file',
    default='../data/antseq.npy',
)

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
seq_file = args.seq_file

seq = np.load(seq_file)

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
        plt.imshow(mask, alpha=0.6, cmap='Blues')
        plt.axis('off')
        plt.savefig('C:/D/CMU/Courses/F23/16-820/hw2/submission/results/ant_{}'.format(i))
