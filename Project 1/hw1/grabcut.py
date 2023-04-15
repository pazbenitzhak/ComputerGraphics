import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import igraph as ig

# Constatns
GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel
CLUSTERS_NUM = 5 # Based on the instructions
GAMMA = 50 # According to the article 
LAST_ENERGY = None
CONVERGENCE_ITERATIONS = -1
EPSILON = 0.001
CONVERGENCE_NUM = 5
N_EDGES = []
N_CAPACITIES = []
GRAPH_SOURCE = -1   # s vertex
GRAPH_SINK = -1     # t vertex
N_LINK_CALC = False
K_CALC = -1

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
    else:
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
    beta = calc_beta(img)
    # We wright it like this so each node will have a unique identifier in the graph
    img_indicies = np.arange(img.shape[0] * img.shape[1])
    img_indicies.reshape(img.shape[0], img.shape[1])

    # We calculating both N(m,n) for n-links and K for t-links for a later use
    k_list = np.array([0 for i in range(img.shape[0]*img.shape[1])])
    zero_row = np.array([0 for i in range(img.shape[1])])

    # Upper left neighbor
    nodes = img_indicies[1:,1:]
    neighbors = img_indicies[:-1,:-1]
    edges.extend(list(zip(nodes, neighbors)))
    
    # N(m,n) function
    temp_capacity = (50 / np.sqrt(2))* \
        (np.exp(-beta*np.square((np.linalg.norm(nodes - neighbors)))))
    
    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())
    upper_left = np.copy(temp_capacity)
    upper_left = np.insert(upper_left, 0, 0, axis=1)
    upper_left = upper_left.reshape(-1)
    upper_left = np.concatenate(zero_row[1:], upper_left)
    down_right = np.copy(temp_capacity)
    down_right = np.insert(down_right, 0, down_right.shape[1], axis=1)
    down_right = down_right.reshape(-1)
    down_right = np.concatenate(down_right, zero_row[1:])
    k_list += upper_left
    k_list += down_right


    # Upper right neighbor
    nodes = img_indicies[1:,:-1]
    neighbors = img_indicies[:-1,1:]
    edges.extend(list(zip(nodes, neighbors)))
    
    # N(m,n) function
    temp_capacity = (50 / np.sqrt(2))* \
        (np.exp(-beta*np.square((np.linalg.norm(nodes - neighbors)))))
    
    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())
    upper_right = np.copy(temp_capacity)
    upper_right = np.insert(upper_right, 0, upper_right.shape[1], axis=1)
    upper_right = upper_right.reshape(-1)
    upper_right = np.concatenate(zero_row[1:], upper_left)
    down_left = np.copy(temp_capacity)
    down_left = np.insert(down_left, 0, 0, axis=1)
    down_left = down_left.reshape(-1)
    down_left = np.concatenate(down_left, zero_row[1:])
    k_list += upper_right
    k_list += down_left
    

    # Upper neighbor
    nodes = img_indicies[1:,:]
    neighbors = img_indicies[:-1,:]
    edges.extend(list(zip(nodes, neighbors)))
    
    # N(m,n) function
    temp_capacity = 50 * \
        (np.exp(-beta*np.square((np.linalg.norm(nodes - neighbors)))))
    
    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())
    upper_concat = np.concatenate((zero_row, one_row_capacity))
    lower_concat = np.concatenate((one_row_capacity, zero_row))
    k_list += upper_concat
    k_list += lower_concat

    # Left neighbor
    nodes = img_indicies[:,1:]
    neighbors = img_indicies[:,:-1]
    edges.extend(list(zip(nodes, neighbors)))
    # N(m,n) function
    temp_capacity = 50 * \
        (np.exp(-beta*np.square((np.linalg.norm(nodes - neighbors)))))
    
    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())
    
    left_concat = np.copy(temp_capacity)
    left_concat = np.insert(left_concat, 0, 0, axis=1)
    right_concat = np.copy(temp_capacity)
    right_concat = np.insert(right_concat, right_concat.shape[1], 0, axis=1)
    k_list += left_concat.reshape(-1)
    k_list += right_concat.reshape(-1)

    # Calculating K vlaue
    K_CALC = max(k_list)

    return edges, capacities



def calc_d_likelihood(img, mask, weights, mean, cov, isBG):
    if isBG == GC_PR_BGD:
        diffrence = img[mask == GC_PR_BGD] - mean
    else:
        diffrence = img[mask == GC_PR_FGD] - mean

    value = weights * 1/np.sqrt((np.linalg.det(cov))) * \
    np.exp(-0.5*np.dot(diffrence.T, np.dot(np.linalg.inv(cov), diffrence)))
    return value
    

def add_t_links(edges, capacities, img, mask, bgGMM, fgGMM, bg_indicies, fg_indicies, unk_indicies):
    # The source is our foreground node, and sink is background node
    # If we got here we already have the K value calculated

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

    # source & unknownBG
    edges.extend(list(zip([GRAPH_SOURCE] * unk_indicies.size, unk_indicies)))
    for i in range(CLUSTERS_NUM):
        background_val += calc_d_likelihood(img, mask, bgGMM[0], bgGMM[1], bgGMM[2], GC_PR_BGD)
    total_value = -np.log10(background_val)
    capacities.extend(total_value.tolist())

    # sink & unknownFG
    edges.extend(list(zip([GRAPH_SINK] * unk_indicies.size, unk_indicies)))
    for i in range(CLUSTERS_NUM):
        foreground_val = calc_d_likelihood(img, mask, fgGMM[0], fgGMM[1], fgGMM[2], GC_PR_FGD)
    total_value = -np.log10(foreground_val)
    capacities.extend(total_value.tolist())

    """
    Is D func returns scalar or matrix?
    is cov of the matrix is 1X1?
    """


    return edges, capacities


def build_graph(img, mask, bgGMM, fgGMM):
    GRAPH_SOURCE = img.shape[0] * img.shape[1]
    GRAPH_SINK = GRAPH_SOURCE + 1
    bg_indicies = np.where(mask == GC_BGD)
    fg_indicies = np.where(mask == GC_FGD)
    # check for a better implenentation on refrence*****************************************
    unk_indices = np.where(mask == GC_PR_BGD or mask == GC_PR_FGD)

    edges = []
    capacities = []

    # Creating n links
    if not N_LINK_CALC:
        N_EDGES, N_CAPACITIES = add_n_links(edges, capacities, img)
        N_LINK_CALC = True

    edges.extend(N_EDGES)
    capacities.extend(N_CAPACITIES)

    # Creating t links
    t_edges, t_capacities = \
        add_t_links(edges, capacities, img, mask, bgGMM, fgGMM, \
                     bg_indicies[0], fg_indicies[0], unk_indices[0])
    edges.extend(t_edges)
    capacities.extend(t_capacities)

    # Building the graph with s and t additional verticies
    graph = ig.Graph(img.shape[0]*img.shape[1] + 2)
    graph.add_edges(edges)

    return graph, capacities



def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    graph, capacities = build_graph(img, mask, bgGMM, fgGMM)
    st_mincut = graph.st_mincut(GRAPH_SOURCE, GRAPH_SINK, capacities)
    energy = st_mincut[0]
    min_cut = [st_mincut[1], st_mincut[2]]
    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    if CONVERGENCE_ITERATIONS == -1 :
        CONVERGENCE_ITERATIONS += 1
        convergence = False
    
    elif np.abs(energy - LAST_ENERGY) < EPSILON:
        CONVERGENCE_ITERATIONS += 1
        if CONVERGENCE_ITERATIONS == CONVERGENCE_NUM:
            convergence = True
    else:
        CONVERGENCE_ITERATIONS = 0
        convergence = False
        LAST_ENERGY = energy
        
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
    fg_intersection = (predicted_mask == gt_mask and predicted_mask != GC_BGD).astype(int)
    intersection_size = np.count_nonzero(fg_intersection)
    jaccard = intersection_size / (intersection_size + (total_pixels - cnt))
    
    return accuarcy, jaccard

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
