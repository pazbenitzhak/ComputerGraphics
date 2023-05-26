import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import igraph as ig

# Constatns
GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel
CLUSTERS_NUM = 8 # Based on the instructions
LAST_ENERGY = 0
CONVERGENCE_ITERATIONS = -1
EPSILON = 0.0001
CONVERGENCE_NUM = 2
N_EDGES = []
N_CAPACITIES = []
GRAPH_SOURCE = -1   # s vertex
GRAPH_SINK = -1     # t vertex
N_LINK_CALC = False
K_CALC = -1
LAST_CUT_SIZE = 0
DIFF_CUT = -1
IS_INITAL = False

# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    # img = img.astype(np.float32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD
    

    bgGMM, fgGMM = initalize_GMMs(img, mask)
    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask, img)
        
        if check_convergence(energy):
            mask[GC_PR_BGD] = GC_BGD
            #mask[GC_PR_FGD] = GC_FGD
            
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def get_triplets(indices,labels,comps):
    weights = []
    means = []
    covs = [] 

    for i in range(comps):
        sliced_img = indices[labels == i]
        if sliced_img.size < 9:
            means.append(np.array([-1, -1, -1]))
            cov = EPSILON*np.eye(3)
            covs.append(cov)
        else:
            means.append(np.mean(sliced_img, axis=0))
            cov = EPSILON*np.eye(3) + np.cov(sliced_img.T)
            covs.append(cov)

    means = np.array(means)
    covs = np.array(covs)
    weights = np.ones(comps) / comps
    pre_chol = np.linalg.cholesky(np.linalg.inv(covs)).transpose((0, 2, 1))
    return weights, means, covs, pre_chol


# NEEDS TO ADD n components=5
def initalize_GMMs(img, mask):
    # TODO: implement initalize_GMMs
    BGkmeans = KMeans(CLUSTERS_NUM).fit(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)])
    FGkmeans = KMeans(CLUSTERS_NUM).fit(img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)])
    bgGMM = GaussianMixture(CLUSTERS_NUM, random_state=0)
    fgGMM = GaussianMixture(CLUSTERS_NUM, random_state=0)
    bg_weights, bg_means, bg_covs, bg_prechol = get_triplets(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)], BGkmeans.labels_, CLUSTERS_NUM)
    bgGMM.weights_ = bg_weights
    bgGMM.means_ = bg_means
    bgGMM.covariances_ = bg_covs
    bgGMM.precisions_cholesky_ = bg_prechol

    fg_weights, fg_means, fg_covs, fg_prechol = get_triplets(img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)], FGkmeans.labels_, CLUSTERS_NUM)
    fgGMM.weights_ = fg_weights
    fgGMM.means_ = fg_means
    fgGMM.covariances_ = fg_covs
    fgGMM.precisions_cholesky_ = fg_prechol

    
    return bgGMM, fgGMM



# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    global IS_INITAL

    h_img = img.shape[0]
    w_img = img.shape[1]
    

    new_img = np.argwhere(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)])
    new_img = np.argwhere(np.logical_or(mask == GC_BGD, mask == GC_PR_BGD))
    flat_new_img = np.ravel_multi_index(np.transpose(new_img),(h_img, w_img))
    indices = img.reshape((img.shape[0] * img.shape[1]), img.shape[2])[flat_new_img]
    labels = bgGMM.predict(indices)
    bg_weights, bg_means ,bg_covs, bg_pre_chol = get_triplets(indices, labels, CLUSTERS_NUM)
    bg_gmm = GaussianMixture(CLUSTERS_NUM)
    bg_gmm.weights_ = bg_weights
    bg_gmm.means_ = bg_means
    bg_gmm.covariances_ = bg_covs
    bg_gmm.precisions_cholesky_ = bg_pre_chol
    
    new_img = np.argwhere(img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)])
    new_img = np.argwhere(np.logical_or(mask == GC_FGD, mask == GC_PR_FGD))
    flat_new_img = np.ravel_multi_index(np.transpose(new_img), (h_img, w_img))
    indices = img.reshape((img.shape[0] * img.shape[1]), img.shape[2])[flat_new_img]
    labels = fgGMM.predict(indices)
    fg_weights, fg_means, fg_covs, fg_pre_chol = get_triplets(indices, labels, CLUSTERS_NUM)
    fg_gmm = GaussianMixture(CLUSTERS_NUM)
    fg_gmm.weights_ = fg_weights
    fg_gmm.means_ = fg_means
    fg_gmm.covariances_ = fg_covs
    fg_gmm.precisions_cholesky_ = fg_pre_chol


    return bg_gmm, fg_gmm


def calc_beta(img):
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
    denomenator = (4 * img.shape[0] * img.shape[1]) - (3 * img.shape[1]) - \
          (3 * img.shape[0]) + 2
    
    
    expectation = nomenator / denomenator
    return 1 / (2 * expectation)


def add_n_links(edges, capacities, img):
    global K_CALC
    beta = calc_beta(img)

    # We wright it like this so each node will have a unique identifier in the graph
    img_indicies = np.arange(img.shape[0] * img.shape[1])
    img_indicies = img_indicies.reshape(img.shape[0], img.shape[1])

    # We calculating both N(m,n) for n-links and K for t-links for a later use
    zero_row = np.array([0 for i in range(img.shape[1])])

    # Upper left neighbor
    nodes = img_indicies[1:,1:]
    neighbors = img_indicies[:-1,:-1]
    nodes_reshape = nodes.reshape(-1)
    neighbors_reshape = neighbors.reshape(-1)
    edges.extend(list(zip(nodes_reshape, neighbors_reshape)))
    

    # N(m,n) function
    temp_capacity = (50 / np.sqrt(2)) * \
        (np.exp(-beta*np.sum(np.square(img[1:, 1:] - img[:-1, :-1]), axis=2)))
    
  
    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())


    # Upper right neighbor
    nodes = img_indicies[1:,:-1]
    neighbors = img_indicies[:-1,1:]
    nodes_reshape = nodes.reshape(-1)
    neighbors_reshape = neighbors.reshape(-1)
    edges.extend(list(zip(nodes_reshape, neighbors_reshape)))
    
    # N(m,n) function
    temp_capacity = (50 / np.sqrt(2)) * \
        (np.exp(-beta*np.sum(np.square(img[1:, :-1] - img[:-1, 1:]), axis=2)))
    

    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())
    upper_right = np.copy(temp_capacity)
    upper_right = np.insert(upper_right, 0, upper_right.shape[1], axis=1)
    upper_right = upper_right.reshape(-1)
    upper_right = np.concatenate((zero_row, upper_right), axis=0)
    down_left = np.copy(temp_capacity)
    down_left = np.insert(down_left, 0, 0, axis=1)
    down_left = down_left.reshape(-1)
    down_left = np.concatenate((down_left, zero_row), axis=0)

    

    # Upper neighbor
    nodes = img_indicies[1:,:]
    neighbors = img_indicies[:-1,:]
    nodes_reshape = nodes.reshape(-1)
    neighbors_reshape = neighbors.reshape(-1)
    edges.extend(list(zip(nodes_reshape, neighbors_reshape)))
    
    # N(m,n) function
    temp_capacity = (50 / 1) * \
        (np.exp(-beta*np.sum(np.square(img[1:,:] - img[:-1,:]), axis=2)))
    


    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())
    

    # Left neighbor
    nodes = img_indicies[:,1:]
    neighbors = img_indicies[:,:-1]
    nodes_reshape = nodes.reshape(-1)
    neighbors_reshape = neighbors.reshape(-1)
    edges.extend(list(zip(nodes_reshape, neighbors_reshape)))
    # N(m,n) function
    temp_capacity = (50 / 1) * \
        (np.exp(-beta*np.sum(np.square(img[:,1:] - img[:,:-1]), axis=2)))
    

    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())

    #k_list += left_concat.reshape(-1)
    #k_list += right_concat.reshape(-1)

    #create graph
    n_graph = ig.Graph(img.shape[0]*img.shape[1])
    n_graph.add_edges(edges)
    n_graph.es['weight'] = capacities
    stregnths = n_graph.strength(weights='weight', mode='in')

    # Calculating K vlaue
    K_CALC = max(stregnths)

    return edges, capacities



def calc_d_likelihood(new_img, means, covs):
    difference = new_img - means
    likelihood = np.sum(np.multiply(difference @ np.linalg.inv(covs), difference), axis=1)
    return likelihood


def add_t_links(edges, capacities, img, mask, bgGMM, fgGMM, bg_indicies, fg_indicies, unk_indicies):
    # The source is our foreground node, and sink is background node
    # If we got here we already have the K value calculated
    global N_LINK_CALC, GRAPH_SINK, GRAPH_SOURCE, K_CALC
    # source & background
    edges.extend(list(zip([GRAPH_SOURCE] * bg_indicies.size, bg_indicies)))
    capacities.extend([0] * bg_indicies.size)

    # sink & background
    edges.extend(list(zip([GRAPH_SINK] * bg_indicies.size, bg_indicies)))
    capacities.extend([K_CALC] * bg_indicies.size)

    # source & foreground
    edges.extend(list(zip([GRAPH_SOURCE] * fg_indicies.size, fg_indicies)))
    capacities.extend([K_CALC] * fg_indicies.size)

    # sink & foreground
    edges.extend(list(zip([GRAPH_SINK] * fg_indicies.size, fg_indicies)))
    capacities.extend([0] * fg_indicies.size)
    flat_img = img.reshape(-1, 3)
    #new_img = flat_img[unk_indicies]
    flat_mask = mask.flatten()

    
    new_img = flat_img[np.where(np.logical_or(flat_mask == GC_PR_BGD, flat_mask == GC_PR_FGD))[0]]
    
    # source & unknown
    edges.extend(list(zip([GRAPH_SOURCE] * unk_indicies.size, unk_indicies)))
    lst_values = []
    for i in range(CLUSTERS_NUM):
        
        lst_values.append(calc_d_likelihood(new_img, bgGMM.means_[i], bgGMM.covariances_[i])) 
    
    values = np.exp(-0.5 * np.array(lst_values))
    root = bgGMM.weights_ / np.sqrt(np.linalg.det(bgGMM.covariances_))
    result = -np.log(root @ values)
    capacities.extend(result.tolist())

    
    # sink & unknown
    edges.extend(list(zip([GRAPH_SINK] * unk_indicies.size, unk_indicies)))
    lst_values = []
    for i in range(CLUSTERS_NUM):
        lst_values.append(calc_d_likelihood(new_img, fgGMM.means_[i], fgGMM.covariances_[i]))
    
    values = np.exp(-0.5 * np.array(lst_values))
    root = fgGMM.weights_ / np.sqrt(np.linalg.det(fgGMM.covariances_))
    result = -np.log(root @ values)
    

    capacities.extend(result.tolist())


    return edges, capacities




def build_graph(img, mask, bgGMM, fgGMM):
    global N_LINK_CALC, GRAPH_SINK, GRAPH_SOURCE, N_CAPACITIES, N_EDGES
    GRAPH_SOURCE = img.shape[0] * img.shape[1]
    GRAPH_SINK = GRAPH_SOURCE + 1
    flat_mask = mask.flatten()
    bg_indicies = np.where(mask == GC_BGD)
    fg_indicies = np.where(mask == GC_FGD)
    unk_indices = np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))


    edges = []
    capacities = []

    n_calc = N_LINK_CALC
    # Creating n links
    if not n_calc:
        add_n_links(edges, capacities, img)
        N_EDGES = edges.copy()
        N_CAPACITIES = capacities.copy()
        N_LINK_CALC = True
    else:
        edges.extend(N_EDGES)
        capacities.extend(N_CAPACITIES)

    # Creating t links
    t_edges, t_capacities = add_t_links(edges, capacities, img, mask, bgGMM, fgGMM, bg_indicies[0], fg_indicies[0], unk_indices[0])



    c_edges = t_edges.copy()
    # Building the graph with s and t additional verticies
    graph = ig.Graph(img.shape[0]*img.shape[1] + 2)
    graph.add_edges(c_edges)
    t_edges.clear()


    return graph, t_capacities



def calculate_mincut(img, mask, bgGMM, fgGMM):
    global GRAPH_SINK, GRAPH_SOURCE, LAST_CUT_SIZE, DIFF_CUT
    # TODO: implement energy (cost) calculation step and mincut
    graph, capacities = build_graph(img, mask, bgGMM, fgGMM)
    c_capacities = capacities.copy()
    st_mincut = graph.st_mincut(GRAPH_SOURCE, GRAPH_SINK, c_capacities)
    capacities.clear()
    energy = st_mincut.value
    min_cut = [st_mincut.partition[0], st_mincut.partition[1]]
    DIFF_CUT = len(st_mincut.partition[0]) - LAST_CUT_SIZE
    LAST_CUT_SIZE = len(st_mincut.partition[0])
    return min_cut, energy


def update_mask(mincut_sets, mask, img):
    # TODO: implement mask update step

    old_mask = mask
    flat_mask = mask.flatten()
    mask_indicies = np.arange(flat_mask.size)
    condition1 = np.logical_and(np.isin(mask_indicies, mincut_sets[0]), flat_mask == GC_PR_FGD)
    condition2 = np.logical_and(np.isin(mask_indicies, mincut_sets[1]), flat_mask == GC_PR_BGD)
    mask = np.where(condition1, GC_PR_BGD, np.where(condition2, GC_PR_FGD, flat_mask)).reshape(mask.shape)

    return mask


def check_convergence(energy):
    global LAST_ENERGY, EPSILON
    # TODO: implement convergence check
    convergence = True
    if np.abs(energy - LAST_ENERGY)/ energy > EPSILON :
        LAST_ENERGY = energy
        convergence = False
        return convergence

    return convergence



def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation
    """
    our accourcy is calculated by the number of pixels that the predicted mask and gt_mask agree on.
    And then we count them and devide them by the number of total pixels

    """
    compare = (predicted_mask == gt_mask).astype(int)
    cnt = np.count_nonzero(compare)
    total_pixels = predicted_mask.shape[0] * predicted_mask.shape[1]
    accuarcy = cnt / total_pixels

    """
    Our Jaccard is calculted by the number of pixels that they agree on and are not labels 
    as background. Then we count them and devide them all by number of intersection pixels and 
    all agreed pixels.

    """
    fg_intersection = ((predicted_mask == gt_mask) & (predicted_mask != GC_BGD)).astype(int)
    intersection_size = np.count_nonzero(fg_intersection)
    jaccard = intersection_size / (intersection_size + (total_pixels - cnt))
    
    return accuarcy, jaccard

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=str, default=1, help='Read rect from course files')
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
    #img = cv2.GaussianBlur(img, (5, 5), 0)
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


