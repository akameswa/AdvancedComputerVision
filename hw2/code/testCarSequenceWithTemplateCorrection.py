import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load('data/carseq.npy')
rect = [59, 116, 145, 151]
rects = [rect]


It0 = seq[:,:,0]
It = seq[:,:,0]

for i in range(seq.shape[2] - 1):  
    It1 = seq[:,:,i+1]

    p = LucasKanade(It, It1, rect, threshold, num_iters)
    pn = np.array(rects[-1][:2]) - np.array(rects[0][:2]) + p
    p_star = LucasKanade(It0, It1, rects[0], threshold, num_iters, pn)

    if(np.linalg.norm(p_star - pn) <= template_threshold):
        It = seq[:,:,i+1]
        p_star = np.array(rects[0][:2]) - np.array(rects[-1][:2]) + p_star
        p = p_star
    else:
        p = p

    rect = rect + np.array([p[0], p[1], p[0], p[1]])
    rects = np.vstack((rects, rect))

    if(i == 1 or i == 100 or i == 200 or i == 300 or i == 400):
        plt.figure(figsize=(10, 10))
        plt.imshow(It1, cmap='gray')
        plt.axis('off')
        patch = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(patch)
        plt.savefig('C:/D/CMU/Courses/F23/16-820/hw2/submission/results/carseq_wcrt_{}'.format(i))

    np.save('C:/D/CMU/Courses/F23/16-820/hw2/submission/data/carseqrects-wcrt.npy', rects)
