import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel
CLUSTERS_NUM = 5 # Based on the instructions
GAMMA = 50 # According to the article 


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    # w -= x
    # h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

# NEEDS TO ADD n components=5
def initalize_GMMs(img, mask):
    # TODO: implement initalize_GMMs
    BGkmeans = KMeans(CLUSTERS_NUM, random_state=0)
    FGkmeans = KMeans(CLUSTERS_NUM, random_state=0)
    bgGMM = BGkmeans.fit(img[mask==GC_BGD]).labels_
    fgGMM = FGkmeans.fit(img[mask==GC_FGD or mask==GC_PR_FGD]).labels_

    return bgGMM, fgGMM




def d_func(img, mask, weights, mean, cov, isBG):
    if isBG == GC_BGD:
        diffrence = img[mask == GC_BGD] - mean

    diffrence = img[mask == GC_FGD] - mean
    value = -np.log10(weights) + 0.5*np.log10(np.linalg.det(cov)) + \
    0.5*np.dot(diffrence.T, np.dot(np.linalg.inv(cov), diffrence))
    return value
    


def assign_GMMs(img, mask, bgGMM, fgGMM):
    # bgGMM & fgGMM are in kmeans labels
    bg_weights, bg_mean, bg_cov = bgGMM
    fg_weights, fg_mean, fg_cov = fgGMM
    bg_d_values = np.array([])
    fg_d_values = np.array([])
    
    for i in range(CLUSTERS_NUM):
        bg_d_values.append(d_func(img, mask, bg_weights[i], bg_mean[i], bg_cov[i], GC_BGD))
        fg_d_values.append(d_func(img, mask, fg_weights[i], fg_mean[i], fg_cov[i], GC_FGD))

    bg_d_values.T
    fg_d_values.T

    new_bg_assignments = np.argmin(bg_d_values, axis=1)
    new_fg_assignments = np.argmin(fg_d_values, axis=1)

    return new_bg_assignments, new_fg_assignments


def learn_GMMs(img, mask, bgGMM_assign, fgGMM_assign):
    bg_weights = np.bincount(bgGMM_assign) / bgGMM_assign.size
    fg_weights = np.bincount(fgGMM_assign) / fgGMM_assign.size
    bg_mean = np.array([np.mean(img[bgGMM_assign==i] for i in range(CLUSTERS_NUM))])
    fg_mean = np.array([np.mean(img[fgGMM_assign==i] for i in range(CLUSTERS_NUM))])
    bg_cov = np.array([np.cov(img[bgGMM_assign==i] for i in range(CLUSTERS_NUM))])
    fg_cov = np.array([np.cov(img[fgGMM_assign==i] for i in range(CLUSTERS_NUM))])

    bg_param = (bg_weights, bg_mean, bg_cov)
    fg_param = (fg_weights, fg_mean, fg_cov)
    
    return bg_param, fg_param



# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    if not isinstance(bgGMM, tuple):      

        # On first iteration, same fuctionality as learn GMMs
        bg_param, fg_param = learn_GMMs(img, mask, bgGMM, fgGMM)

        new_bg_assign, new_fg_assign = assign_GMMs(img, mask, bg_param, fg_param)
    else:
        new_bg_assign, new_fg_assign = assign_GMMs(img, mask, bgGMM, fgGMM)

    bgGMM, fgGMM = learn_GMMs(img, mask, new_bg_assign, new_fg_assign)

    return bgGMM, fgGMM


def beta(img):
    #in order to avoid duoble calculation we calculate based on 4 way connectivity
    left_neighbor_dist = np.square(img[:, 1:] - img[:, :-1])
    upleft_neighbor_dist = np.square(img[1:, 1:] - img[:-1, :-1])
    up_neighbor_dist = np.square(img[1:, :] - img[:-1, :])
    upright_neighbor_dist = np.square(img[1:, :-1] - img[:-1, 1:])


    nomenator = np.sum(left_neighbor_dist) + np.sum(upleft_neighbor_dist) + \
    np.sum(up_neighbor_dist) + np.sum(upright_neighbor_dist)


    # We need the size of all sub images, basically they are the size of 4 regualer images minus 
    # 3 cloumns because of left, upleft and upright neighbors and minus 3 rows because of upleft,
    # upright and up neighbors. we add 2 more pixels because of double subtruction of the edge
    # pixels of the first row.
    denomenator = (4 * img.shape[0] * img.shape[1]) - (3 * img.shape[1]) - (3 * img.shape[0]) + 2
    
    
    expectation = nomenator / denomenator
    return 1 / (2 * expectation)


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]
    energy = 0
    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
