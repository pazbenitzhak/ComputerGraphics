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

def new_a_matrix(dimention,rows,cols,indices,mask):
    #dimention is number of rows which is the number of pixels in the mask
    #create
    A = scipy.sparse.eye(dimention,dtype=int)
    #main diagonal
    #target values which are supposed to be ones
    inside_indices_converted = [(x*cols+y,x*cols+y) for (x,y) in indices]
    if ((dimention//2,dimention//2) in inside_indices_converted):
        print("WHYYYYYYYYYY")
    values_4 = np.array([-5 for i in range(len(inside_indices_converted))])
    A += scipy.sparse.coo_matrix((values_4, zip(*inside_indices_converted)), shape=(dimention, dimention), dtype=int)
    #now need to take care of all of the rows with value of -4 in them
    #only need to handle adding of ones, so there's no need to include
    #values whose mask value is zero because they will be zeroed in the equation
    #conversion is done accordingly
    up_indices = [(x*cols+y-1,x*cols+y) for (x,y) in indices if (y-1)>=0 \
                  and mask[x,y-1]!=0]
    down_indices = [(x*cols+y+1,x*cols+y) for (x,y) in indices \
                    if (y+1<rows) and mask[x,y+1]!=0 and x*cols+y+1<dimention]
    left_indices = [(x*cols+y-cols,x*cols+y) for (x,y) in indices if \
                    (x-1)>=0 and mask[x-1,y]!=0 and x*cols+y-cols>=0]
    right_indices = [(x*cols+y+cols,x*cols+y) for (x,y) in indices \
                     if (x+1<cols) and mask[x+1,y]!=0 and x*cols+y-cols<dimention]
    len_values_ones_inside_indices = len(up_indices)+len(down_indices)+len(left_indices)+len(right_indices)
    values_ones_inside_indices = np.array([1 for i in range(len_values_ones_inside_indices)])
    #sum everything up
    indices = np.array(up_indices+down_indices+left_indices+right_indices)
    #build matrix
    A += scipy.sparse.coo_matrix((values_ones_inside_indices, zip(*indices)), shape=(dimention, dimention), dtype=int)
    A = A.tocsr()
    return A

def calc_laplacian_operator(src, index):
    # 𝑠(𝑥+1,𝑦)+𝑠(𝑥−1,𝑦)+𝑠(𝑥,𝑦+1)+𝑠(𝑥,𝑦−1)−4𝑠(𝑥,𝑦)
    i, j = index
    val = -4*src[i,j]
    if (i+1<src.shape[1]):
        val+=src[i+1,j]
    if (i-1>=0):
        val+=src[i-1,j]
    if (j+1<src.shape[0]):
        val+=src[i,j+1]
    if (j-1>=0):
        val+=src[i,j-1]
    return val


def classify_point(index, mask):
    # 0 := outside omega
    # 1 := inside omega
    # 2 := delta omega
    i,j = index
    if (i<0 or j<0 or j>=mask.shape[0] or i>=mask.shape[1]):
        #out of bounds so don't slice on array, return 1 so condition not executed
        return 1
    if mask[index] == 0:
        return 0
    
    for neighbor in neighbors(index):
        # if one of his neighbors is outside, than it has to be delta omega
        if mask[neighbor] == 0:
            return 2

    # else meaning we inside omega
    return 1


def new_calc_B_vector(src, mask, tgt, dimention):
    B = np.array([[0,0,0] for i in range(dimention)])
    #tgt to mask
    offset_x = (-mask.shape[0] + tgt.shape[0])//2
    offset_y = (-mask.shape[1] + tgt.shape[1])//2
    #mask to source
    src_to_mask_offset_x = (-mask.shape[0]+src.shape[0])//2
    src_to_mask_offset_y = (-mask.shape[1]+src.shape[1])//2
    for i in range(dimention):
        mask_index = (i//mask.shape[1],i%mask.shape[1])
        tgt_index = mask_index[0]+offset_x, mask_index[1]+offset_y
        #default case
        B[i][0] = tgt[tgt_index][0]
        B[i][1] = tgt[tgt_index][1]
        B[i][2] = tgt[tgt_index][2]
        if (mask[mask_index]==0): 
            #it's outside the mask, therefore it's well defined
            continue
        #else need to do the regular process, including the laplacian. First we start from zero
        B[i][0] = 0
        B[i][1] = 0
        B[i][2] = 0
        src_index = mask_index[0]+src_to_mask_offset_x, mask_index[1]+src_to_mask_offset_y
        val = calc_laplacian_operator(src, src_index)
        B[i][0] += val[0]
        B[i][1] += val[1]
        B[i][2] += val[2]
        # 0 := outside omega
        # 1 := inside omega
        # 2 := delta omega
        if classify_point(mask_index, mask) == 2:
            #for each neighbor substract the target pixel
            for neighbor in neighbors(mask_index):
                if classify_point(neighbor, mask) == 0: #it's outside basically
                    B[i][0] -= tgt[tgt_index][0]
                    B[i][1] -= tgt[tgt_index][1]
                    B[i][2] -= tgt[tgt_index][2]

    return B




def poisson_blend(im_src, im_tgt, im_mask, center):
    if im_mask.size > im_tgt.size:
        raise Exception("Invalid input! Target image bigger than image mask")
    split_indices = np.where(im_mask != 0)
    omega_indices = list(zip(split_indices[0], split_indices[1]))
    mat_rows, mat_cols = im_mask.shape[:2]
    dimention = mat_rows*mat_cols
    A_matrix = new_a_matrix(dimention,mat_rows,mat_cols,omega_indices,im_mask)

    B_vector = new_calc_B_vector(im_src, im_mask, im_tgt, dimention)

    im_blend = np.copy(im_tgt)
    # 𝑥=𝐴\b
    x_0 = np.uint8(scipy.sparse.linalg.spsolve(A_matrix, B_vector[:,0],use_umfpack=True))
    x_1 = np.uint8(scipy.sparse.linalg.spsolve(A_matrix, B_vector[:,1],use_umfpack=True))
    x_2 = np.uint8(scipy.sparse.linalg.spsolve(A_matrix, B_vector[:,2],use_umfpack=True))
    x = np.stack([x_0,x_1,x_2], axis=-1)
    x = np.clip(x, 0, 255)
    #need to put x in target image
    offset_x = (-im_mask.shape[0] + im_tgt.shape[0])//2
    offset_y = (-im_mask.shape[1] + im_tgt.shape[1])//2
    for i in range(dimention):
        mask_index = (i//im_mask.shape[1],i%im_mask.shape[1])
        tgt_index = mask_index[0]+offset_x, mask_index[1]+offset_y
        im_blend[tgt_index][0] = x[i][0] 
        im_blend[tgt_index][1] = x[i][1] 
        im_blend[tgt_index][2] = x[i][2] 
    im_blend = np.uint8(im_blend)
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
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
