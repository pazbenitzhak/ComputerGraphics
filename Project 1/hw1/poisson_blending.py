import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def neighbors(index):
    # since we'll check later if the tuple is in existing indices list we don't 
    # handle diffrently out of range points
    i, j = index
    return [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]


def sparse_matrix(indices, dimention):
    A = scipy.sparse.lil_matrix((dimention, dimention))
    for i in range(dimention):
        A[i,i] = -4
        for neighbor in neighbors(indices[i]):
            if neighbor in indices:
                A[neighbor] = 1
    return A


def calc_laplacian_operator(src, index):
    # 𝑠(𝑥+1,𝑦)+𝑠(𝑥−1,𝑦)+𝑠(𝑥,𝑦+1)+𝑠(𝑥,𝑦−1)−4𝑠(𝑥,𝑦)
    i, j = index
    return src[i+1, j] + src[i-1, j] + src[i, j+1] + src[i, j-1]- 4*src[i, j]


def classify_point(index, mask):
    if mask[index] == 0:
        return 0
    
    for neighbor in neighbors(index):
        # if one of his neighbors is outside, than it has to be delta omega
        if mask[neighbor] == 0:
            return 2

    # else meaning we inside omega
    return 1


def calc_B_vector(src, mask, tgt, indices, dimention):
    B = np.zeros(dimention)
    for i in range(dimention):
        B[i] += calc_laplacian_operator(src, indices[i])
        # 0 := outside omega
        # 1 := inside omega
        # 2 := delta omega
        if classify_point(indices[i], mask) == 2:
            for neighbor in neighbors(indices[i]):
                if classify_point(neighbor, mask) != 1:
                    B[i] -= tgt[neighbor]
    return B


def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    if im_mask.size > im_tgt.size:
        raise Exception("Invalid input! Target image bigger than image mask")
    split_indices = np.where(im_mask == 255)
    omega_indices = list(zip(split_indices[0], split_indices[1]))
    dimention = len(omega_indices)
    A_matrix = sparse_matrix(omega_indices, dimention)
    B_vector = calc_B_vector(im_src, im_mask, im_tgt, omega_indices, dimention)
    new_mask = np.zeros(im_mask.shape)
    for i in range(dimention):
        new_mask[omega_indices[i]] = B_vector[i]
    
    # 𝑥=𝐴\b
    x = np.linalg.cg(A_matrix, B_vector)
    
    im_blend = np.copy(im_tgt)
    mask_center = im_mask.shape[0] // 2, im_mask.shape[1] // 2
    for i in range(-im_mask.shape[0] // 2, im_mask.shape[0] // 2 + 1):
        for j in range(-im_mask.shape[1] // 2, im_mask.shape[1] // 2 + 1):
            item = new_mask[mask_center[0] + i, mask_center[1] + j]
            if item != 0:
                im_blend[center[0] + i, center[1] + j] = item

    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
