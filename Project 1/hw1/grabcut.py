import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
import igraph as ig

# Constatns
GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel
CLUSTERS_NUM = 5 # Based on the instructions
LAST_ENERGY = 0
CONVERGENCE_ITERATIONS = -1
EPSILON = 0.001
CONVERGENCE_NUM = 5
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
    img = img.astype(np.float32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD
    print("number of edges : " +str(2 * (img.shape[0]-1) * img.shape[1] + 2 * img.shape[0] * (img.shape[1]-1)))
    print("number of t links : " +str(img.shape[0] * img.shape[1] ))

    bgGMM, fgGMM = initalize_GMMs(img, mask)
    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        print("_________________________________\n")
        print("number of iteration: " + str(i+1))
        print("_________________________________\n")
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask, img)

        if check_convergence(energy):
            #print("energy : " + str(energy))
            mask[GC_PR_BGD] = GC_BGD
            mask[GC_PR_FGD] = GC_FGD
            
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

# NEEDS TO ADD n components=5
def initalize_GMMs(img, mask):
    # TODO: implement initalize_GMMs
    BGkmeans = KMeans(5)
    FGkmeans = KMeans(5)
    bgkmeans = BGkmeans.fit(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)])
    fgkmeans = FGkmeans.fit(img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)])
    bgGMM_labels = bgkmeans.labels_
    fgGMM_labels = fgkmeans.labels_
    bgGMM_mean = bgkmeans.cluster_centers_
    fgGMM_mean = fgkmeans.cluster_centers_

    bg_weights = np.bincount(bgGMM_labels) / bgGMM_labels.size
    fg_weights = np.bincount(fgGMM_labels) / fgGMM_labels.size
    img_bg = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    img_fg = img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]
    bg_cov = []
    fg_cov = []
    for i in range(CLUSTERS_NUM):
        # m = i - 5
        # bg_mean.append([np.mean(img[new_img == i], axis = 0)])
        # fg_mean.append([np.mean(img[new_img == m], axis = 0)])
        bg_cov.append([np.cov((img_bg[bgGMM_labels == i].T))])
        fg_cov.append([np.cov((img_fg[fgGMM_labels == i].T))])

    bg_param = (bg_weights, bgGMM_mean, bg_cov)
    fg_param = (fg_weights, fgGMM_mean, fg_cov)
    print("bg_param kmeans: " +str(bg_param))
    print("fg_param kmeans: " +str(fg_param))
    return bg_param, fg_param




def d_func(img, mask, weights, mean, cov, isBG):
    cov = cov[0]
    if isBG == GC_BGD:
        diffrence = img[mask == GC_BGD] - mean
    else:
        diffrence = img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)] - mean


    det_of_cov = np.linalg.det(cov)
    if det_of_cov == 0:
        diag = np.diag(cov)
        diag = diag + 0.00001
        np.fill_diagonal(cov, diag)
        value = weights * (1 / np.sqrt(det_of_cov)) * np.exp(-0.5*np.einsum('ij,ij->i', np.dot(diffrence, np.linalg.inv(cov)), diffrence))
    else:    
        value = weights * (1 / np.sqrt(det_of_cov)) * np.exp(-0.5*np.einsum('ij,ij->i', np.dot(diffrence, np.linalg.inv(cov)), diffrence))
    return value
    


def assign_GMMs(img, mask, bgGMM, fgGMM):
    # bgGMM & fgGMM are in kmeans labels
    
    print("inside bg_gmm: " +str(np.shape(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)])))
    print("inside fg_gmm: " +str(np.shape(img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)])))
    gmm = GaussianMixture(5, init_params='kmeans')
    gmm.fit(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)])
    bg_labels = gmm.predict(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)])
    bg_mean = gmm.means_

    gmm = GaussianMixture(5, init_params='kmeans')
    gmm.fit(img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)])
    fg_labels = gmm.predict(img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)])
    fg_mean = gmm.means_

    # bg_weights, bg_mean, bg_cov = bgGMM
    # fg_weights, fg_mean, fg_cov = fgGMM
    # bg_d_values = []
    # fg_d_values = []

    # for i in range(CLUSTERS_NUM):
    #     d_bg = d_func(img, mask, bg_weights[i], bg_mean[i], bg_cov[i], GC_BGD)
    #     bg_d_values.append(d_bg)
    #     d_fg = d_func(img, mask, fg_weights[i], fg_mean[i], fg_cov[i], GC_FGD)
    #     fg_d_values.append(d_fg)

    # print("value : " +str(np.unique(-np.log10(np.sum(bg_d_values, axis=1)))))
    # bg_d_values = np.array(bg_d_values).T
    # fg_d_values = np.array(fg_d_values).T
    
    # new_bg_assignments = np.argmax(bg_d_values, axis=1)
    # new_fg_assignments = np.argmax(fg_d_values, axis=1)

    bgGMM = (bg_labels, bg_mean)
    fgGMM = (fg_labels, fg_mean)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11111")
    return bgGMM, fgGMM
    #return new_bg_assignments, new_fg_assignments


def learn_GMMs(img, mask, bgGMM_assign, fgGMM_assign):
    bg_gmm_labels = bgGMM_assign[0]
    fg_gmm_labels = fgGMM_assign[0]
    bg_weights = np.bincount(bg_gmm_labels) / bg_gmm_labels.size
    fg_weights = np.bincount(fg_gmm_labels) / fg_gmm_labels.size
    # new_img = np.zeros_like(mask, dtype=np.int32)
    # new_img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)] = bg_gmm_labels
    #fg_gmm_labels -= 5
    # new_img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)] = fg_gmm_labels
    img_bg = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    img_fg = img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]
    bg_cov = []
    fg_cov = []
    bg_mean = bgGMM_assign[1]
    fg_mean = fgGMM_assign[1]
    for i in range(CLUSTERS_NUM):
        #m = i - 5
        # bg_mean.append([np.mean(img[new_img == i], axis = 0)])
        # fg_mean.append([np.mean(img[new_img == m], axis = 0)])
        bg_cov.append([np.cov((img_bg[bg_gmm_labels == i].T))])
        fg_cov.append([np.cov((img_fg[fg_gmm_labels == i].T))])

    bg_param = (bg_weights, bg_mean, bg_cov)
    fg_param = (fg_weights, fg_mean, fg_cov)
    
    print("learn GMM !!!!!!!!!!!!!!!!!!!!!!!!!1")
    return bg_param, fg_param



# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    global IS_INITAL
    # TODO: implement GMM component assignment step

    bg_gmm = GaussianMixture(5, init_params='kmeans')
    bg_gmm.weights_ = bgGMM[0]
    bg_gmm.means_ = bgGMM[1]
    bg_gmm.covariances_ = bgGMM[2]
    print("bg_gmm.covariances_ shape: " +str(np.shape(bg_gmm.covariances_)))
    bg_gmm.precisions_cholesky_ = np.array([cholesky(np.linalg.inv(cov[0])) for cov in bg_gmm.covariances_])
    bg_labels = bg_gmm.predict(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)])
    bg_mean = bg_gmm.means_
    bg_cov = bg_gmm.covariances_
    bg_weights = bg_gmm.weights_
    print("cov shape: " +str(np.shape(bg_cov)))
    print("mean shape: " +str(np.shape(bg_mean)))
    print("weights shape: " +str(np.shape(bg_weights)))
    fg_gmm = GaussianMixture(5, init_params='kmeans')
    fg_gmm.weights_ = fgGMM[0]
    fg_gmm.means_ = fgGMM[1]
    fg_gmm.covariances_ = fgGMM[2]
    fg_gmm.precisions_cholesky_ = np.array([cholesky(np.linalg.inv(cov[0])) for cov in fg_gmm.covariances_])
    fg_labels = fg_gmm.predict(img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)])
    fg_mean = fg_gmm.means_
    fg_cov = fg_gmm.covariances_
    fg_weights = fg_gmm.weights_

    # if IS_INITAL:
    #     bgGMM, fgGMM = assign_GMMs(img, mask, bgGMM, fgGMM)
    
    # IS_INITAL = True
    # # if not isinstance(bgGMM, tuple):      
    # #     # On first iteration, same fuctionality as learn GMMs
    # #     bg_param, fg_param = learn_GMMs(img, mask, bgGMM, fgGMM)

    # #     new_bg_assign, new_fg_assign = assign_GMMs(img, mask, bg_param, fg_param)
    # # else:
    # #     new_bg_assign, new_fg_assign = assign_GMMs(img, mask, bgGMM, fgGMM)

    # bgGMM, fgGMM = learn_GMMs(img, mask, bgGMM, fgGMM)
    # print("bgGMMs: " +str(bgGMM))
    # print("fgGMMs: " +str(fgGMM))
    bgGMM = (bg_weights, bg_mean, bg_cov)
    fgGMM = (fg_weights, fg_mean, fg_cov)
    print("update GMM !!!!!!!!!!!!!!!!!!!!!!!!!1")
    print("bg_param GMMs: " +str(bgGMM))
    print("fg_param GMMs: " +str(fgGMM))
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
    global K_CALC
    beta = calc_beta(img)
    print("beta: " + str(beta))
    # We wright it like this so each node will have a unique identifier in the graph
    img_indicies = np.arange(img.shape[0] * img.shape[1])
    img_indicies = img_indicies.reshape(img.shape[0], img.shape[1])

    # We calculating both N(m,n) for n-links and K for t-links for a later use
    #k_list = np.array([0.0 for i in range(img.shape[0]*img.shape[1])])
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
    
    # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
    # print("temp_capacity 1: " + str(np.unique(temp_capacity)))
    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())
    print("capacites size: " + str(len(capacities)))
    print("edges size: " + str(len(edges)))
    #print("upper_left: " + str(np.unique(upper_left)))
    #print("-------------------------\n")
    #print("-------------------------\n")
    # k_list += upper_left
    # k_list += down_right


    # Upper right neighbor
    nodes = img_indicies[1:,:-1]
    neighbors = img_indicies[:-1,1:]
    nodes_reshape = nodes.reshape(-1)
    neighbors_reshape = neighbors.reshape(-1)
    edges.extend(list(zip(nodes_reshape, neighbors_reshape)))
    
    # N(m,n) function
    temp_capacity = (50 / np.sqrt(2)) * \
        (np.exp(-beta*np.sum(np.square(img[1:, :-1] - img[:-1, 1:]), axis=2)))
    
    # print("temp_capacity 2: " + str(np.unique(temp_capacity)))

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
    #print("upper_right: " + str(np.unique(upper_right)))
    #print("-------------------------\n")
    #print("down_left: " + str(np.unique(down_left)))
    #print("-------------------------\n")
    # k_list += upper_right
    # k_list += down_left
    

    # Upper neighbor
    nodes = img_indicies[1:,:]
    neighbors = img_indicies[:-1,:]
    nodes_reshape = nodes.reshape(-1)
    neighbors_reshape = neighbors.reshape(-1)
    edges.extend(list(zip(nodes_reshape, neighbors_reshape)))
    
    # N(m,n) function
    temp_capacity = (50 / 1) * \
        (np.exp(-beta*np.sum(np.square(img[1:,:] - img[:-1,:]), axis=2)))
    
    print("temp_capacity 3: " + str(np.unique(temp_capacity)))


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
    
    print("temp_capacity 4: " + str(np.unique(temp_capacity)))

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
    one_row_capacity = temp_capacity.reshape(-1)
    capacities.extend(one_row_capacity.tolist())
    print("capacites size: " + str(len(capacities)))
    print("edges size: " + str(len(edges)))
    print("#######################################################\n")
    
    #k_list += left_concat.reshape(-1)
    #k_list += right_concat.reshape(-1)

    #create graph
    n_graph = ig.Graph(img.shape[0]*img.shape[1])
    n_graph.add_edges(edges)
    n_graph.es['weight'] = capacities
    stregnths = n_graph.strength(weights='weight', mode='in')

    # Calculating K vlaue
    #print("k_list : " + str(k_list))
    print("------------------------- \n")
    #K_CALC = max(k_list)
    K_CALC = max(stregnths)
    print("K_CALC value: " + str(K_CALC))
    return edges, capacities



def calc_d_likelihood(img, mask, weights, mean, cov, isBG):
    # only Unk indices caling this method
    #print("cov shape : " + str(np.shape(cov)))
    cov = cov[0]
    #difference = img - mean
    value = multivariate_normal.pdf(img, mean=mean, cov=cov)
    #value = np.sum(np.multiply(difference @ cov, difference), axis=1)
    # if isBG == GC_PR_BGD:
    #     diffrence = img[mask == GC_PR_BGD] - mean
    # else:
    #     diffrence = img[mask == GC_PR_FGD] - mean
    # det_of_cov = np.linalg.det(cov)
    # if det_of_cov == 0:
    #     diag = np.diag(cov)
    #     diag = diag + 0.00001
    #     np.fill_diagonal(cov, diag)
    #     value = weights * 1/np.sqrt((np.linalg.det(cov))) * \
    #     np.exp(-0.5*np.einsum('ijk,ijk->ij', np.dot( difference,np.linalg.inv(cov)), difference))
    #     #np.exp(-0.5 * np.sum(np.matmul(difference, np.linalg.inv(cov)), axis=1))
    # else:
    #     value = weights * 1/np.sqrt((np.linalg.det(cov))) * \
    #     np.exp(-0.5*np.einsum('ijk,ijk->ij', np.dot( difference,np.linalg.inv(cov)), difference))
        # np.exp(-0.5 * np.sum(np.matmul(difference, np.linalg.inv(cov)), axis=1))
    # np.exp(-0.5*np.einsum('ijk,ijk->ij', np.dot( difference,np.linalg.inv(cov)), difference))
    #print("value shape: " +str(np.shape(value)))
    return weights*value 
    

def add_t_links(edges, capacities, img, mask, bgGMM, fgGMM, bg_indicies, fg_indicies, unk_indicies):
    # The source is our foreground node, and sink is background node
    # If we got here we already have the K value calculated
    global N_LINK_CALC, GRAPH_SINK, GRAPH_SOURCE, K_CALC
    # source & background
    edges.extend(list(zip([GRAPH_SOURCE] * bg_indicies.size, bg_indicies)))
    capacities.extend([0] * bg_indicies.size)
    #print("source & background : " + str(len(list(zip([GRAPH_SOURCE] * bg_indicies.size, bg_indicies)))))

    # sink & background
    edges.extend(list(zip([GRAPH_SINK] * bg_indicies.size, bg_indicies)))
    capacities.extend([K_CALC] * bg_indicies.size)
    #print("sink & background : " + str(len(list(zip([GRAPH_SINK] * bg_indicies.size, bg_indicies)))))

    # source & foreground
    edges.extend(list(zip([GRAPH_SOURCE] * fg_indicies.size, fg_indicies)))
    capacities.extend([K_CALC] * fg_indicies.size)
    #print("source & foreground : " + str(len(list(zip([GRAPH_SOURCE] * fg_indicies.size, fg_indicies)))))

    # sink & foreground
    edges.extend(list(zip([GRAPH_SINK] * fg_indicies.size, fg_indicies)))
    capacities.extend([0] * fg_indicies.size)
    #flat_img = img.reshape(-1, 3)
    # new_img = flat_img[unk_indicies]
    #print("sink & foreground : " + str(len(list(zip([GRAPH_SINK] * fg_indicies.size, fg_indicies)))))

    
    new_img = img[np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD)]
    
    # source & unknown
    edges.extend(list(zip([GRAPH_SOURCE] * unk_indicies.size, unk_indicies)))
    lst_values = np.zeros(shape = new_img.shape[:1])
    #print("source & unknown : " + str(len(list(zip([GRAPH_SOURCE] * unk_indicies.size, unk_indicies)))))


    for i in range(CLUSTERS_NUM):
        # print("background_val number " + str(i))
        # print("bgGMM[0][i] : " + str(bgGMM[0][i]))
        # print("bgGMM[1][i] : " + str(bgGMM[1][i]))
        # print("bgGMM[2][i] : " + str(bgGMM[2][i]))
        lst_values += calc_d_likelihood(new_img, mask, bgGMM[0][i], bgGMM[1][i], bgGMM[2][i], GC_PR_BGD)
    #values = np.exp(-0.5 * np.array(lst_values))
    # print("--------------------------------------------\n")
    # print("list values: " +str(lst_values))
    # print("inside the exp: " + str(-0.5 * np.array(lst_values)))
    # print("values: " + str(values))
    # print("--------------------------------------------\n")
        
    # det_cov = []
    # for i in range(len(bgGMM[2])):
    #     det = np.linalg.det(bgGMM[2][i])
    #     if det == 0:
    #         diag = np.diag(bgGMM[2][i])
    #         diag = diag + 0.00001
    #         np.fill_diagonal(bgGMM[2][i], diag)
    #         det = np.linalg.det(bgGMM[2][i])
    #     det_cov.append(det)

    # print("det bgGMM[2]: " + str(1/np.sqrt((np.linalg.det(bgGMM[2])))))
    # print("det bgGMM[0]: " + str(bgGMM[0]))
    # print("values: " + str(values))
    # print("inside the log: " + str(bgGMM[0] * 1/np.sqrt((np.linalg.det(bgGMM[2]))) @ values))
    # print("---------------------------------------------\n")

    #det_cov = np.array(det_cov)
    #                                        we need this and not det_cov?   problem
    #total_value = -np.log((bgGMM[0] * 1/np.sqrt((np.linalg.det(bgGMM[2])))) @ values)
    total_value = -np.log(lst_values)
    capacities.extend(total_value.tolist())

    
    # sink & unknown
    edges.extend(list(zip([GRAPH_SINK] * unk_indicies.size, unk_indicies)))
    lst_values = np.zeros(shape = new_img.shape[:1])
    print("lst_values shape: " + str(np.shape(lst_values)))
    for i in range(CLUSTERS_NUM):
        #print("background_val number " + str(i))
        lst_values += calc_d_likelihood(new_img, mask, fgGMM[0][i], fgGMM[1][i], fgGMM[2][i], GC_PR_FGD)
    # values = np.exp(-0.5 * np.array(lst_values))
    
    # det_cov = []
    # for i in range(len(fgGMM[2])):
    #     det = np.linalg.det(fgGMM[2][i])
    #     if det == 0:
    #         diag = np.diag(fgGMM[2][i])
    #         diag = diag + 0.00001
    #         np.fill_diagonal(fgGMM[2][i], diag)
    #         det = np.linalg.det(fgGMM[2][i])
    #     det_cov.append(det)

    # det_cov = np.array(det_cov)
    # total_value = -np.log((fgGMM[0] * 1/np.sqrt((np.linalg.det(fgGMM[2])))) @ values)
    total_value = -np.log(lst_values)
    print("---------------------\n")
    print("total value: " +str(total_value))

    capacities.extend(total_value.tolist())


    # if unk_fgd_indicies.size > 0:
    #     # source & unknownFG
    #     edges.extend(list(zip([GRAPH_SOURCE] * unk_fgd_indicies.size, unk_fgd_indicies)))
    #     for i in range(CLUSTERS_NUM):
    #         print("foreground_val number " + str(i))
    #         foreground_val = calc_d_likelihood(img, mask, fgGMM[0][i], fgGMM[1][i], fgGMM[2][i], GC_PR_FGD)
    #     total_value = -np.log10(foreground_val)
    #     print("---------------------\n")
    #     print("total value: " +str(np.shape(total_value)))
    #     capacities.extend(total_value.tolist())


    # # sink & unknownFG
    # edges.extend(list(zip([GRAPH_SINK] * unk_fgd_indicies.size, unk_fgd_indicies)))
    # for i in range(CLUSTERS_NUM):
    #     print("foreground_val number " + str(i))
    #     foreground_val = calc_d_likelihood(img, mask, fgGMM[0][i], fgGMM[1][i], fgGMM[2][i], GC_PR_FGD)
    # total_value = -np.log10(foreground_val)
    # capacities.extend(total_value.tolist())

    print("capacites size t links: " + str(len(capacities)))
    print("edges size t links: " + str(len(edges)))
    print("#######################################################\n")

    return edges, capacities


def build_graph(img, mask, bgGMM, fgGMM):
    global N_LINK_CALC, GRAPH_SINK, GRAPH_SOURCE, N_CAPACITIES, N_EDGES
    GRAPH_SOURCE = img.shape[0] * img.shape[1]
    GRAPH_SINK = GRAPH_SOURCE + 1
    flat_mask = mask.flatten()
    bg_indicies = np.where(mask == GC_BGD)
    fg_indicies = np.where(mask == GC_FGD)
    unk_indices = np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))
    #unk_indices = np.where(np.logical_or(flat_mask == GC_PR_BGD, flat_mask == GC_PR_FGD))

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

    # print("N_EDGES LEN: " + str(len(N_EDGES)))
    # print("--------------------------------------\n")
    # print("N_CAPACITIES : " + str(len(N_CAPACITIES)))
    # print("--------------------------------------\n")
    # Creating t links
    t_edges, t_capacities = add_t_links(edges, capacities, img, mask, bgGMM, fgGMM, bg_indicies[0], fg_indicies[0], unk_indices[0])
    #edges.extend(t_edges)
    #capacities.extend(t_capacities)


    # print("--------------------------------------\n")
    # print("capacities2 : " + str(len(capacities)))
    # print("--------------------------------------\n")
    c_edges = t_edges.copy()
    # Building the graph with s and t additional verticies
    graph = ig.Graph(img.shape[0]*img.shape[1] + 2)
    graph.add_edges(c_edges)
    t_edges.clear()

    print("finish building the graph")
    # with open(f'banana1_edge_weight.txt', 'w') as f:
    #     f.write('\n'.join([str(a) for a in zip(edges, capacities)]))

    print("edges : " +str(np.unique(np.array(capacities))))
    return graph, t_capacities



def calculate_mincut(img, mask, bgGMM, fgGMM):
    global GRAPH_SINK, GRAPH_SOURCE, LAST_CUT_SIZE, DIFF_CUT
    # TODO: implement energy (cost) calculation step and mincut
    graph, capacities = build_graph(img, mask, bgGMM, fgGMM)
    c_capacities = capacities.copy()
    st_mincut = graph.st_mincut(GRAPH_SOURCE, GRAPH_SINK, c_capacities)
    capacities.clear()
    print("************************************************\n")
    print("st_mincut : " + str(st_mincut))
    print("************************************************\n")
    energy = st_mincut.value
    min_cut = [st_mincut.partition[0], st_mincut.partition[1]]
    DIFF_CUT = len(st_mincut.partition[0]) - LAST_CUT_SIZE
    LAST_CUT_SIZE = len(st_mincut.partition[0])
    return min_cut, energy


def update_mask(mincut_sets, mask, img):
    # TODO: implement mask update step
    print("updating mask")
    old_mask = mask
    flat_mask = mask.flatten()
    # unk_indices = np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))
    mask_indicies = np.arange(flat_mask.size)
    # #print("array_indices shape: " + str(np.shape(array_indicies)))
    # array_indicies = np.reshape(array_indicies ,(img.shape[0], img.shape[1]))
    condition1 = np.logical_and(np.isin(mask_indicies, mincut_sets[0]), flat_mask == GC_PR_FGD)
    #mask[unk_indices] = np.where(condition1, GC_PR_FGD, GC_PR_BGD)
    condition2 = np.logical_and(np.isin(mask_indicies, mincut_sets[1]), flat_mask == GC_PR_BGD)
    #mask[unk_indices] = np.where(condition2, GC_PR_BGD, GC_PR_FGD)
    mask = np.where(condition1, GC_PR_BGD, np.where(condition2, GC_PR_FGD, flat_mask)).reshape(mask.shape)

    # mask_f = mask.flatten()
    # condition__gc_pr_bgd = np.logical_and(np.isin(np.arange(mask_f.size), mincut_sets[0]), mask_f == GC_PR_FGD)
    # condition__gc_pr_fgd = np.logical_and(np.isin(np.arange(mask_f.size), mincut_sets[1]), mask_f == GC_PR_BGD)
    # mask = np.where(condition__gc_pr_bgd, GC_PR_BGD, np.where(condition__gc_pr_fgd, GC_PR_FGD, mask_f)).reshape(mask.shape)



    print(f'Number of mask changes (prev {np.sum(old_mask)} new {np.sum(mask)}: {np.sum(np.nonzero(old_mask - mask))}')

    return mask


def check_convergence(energy):
    global LAST_ENERGY, EPSILON, DELTA_CUT
    # TODO: implement convergence check
    convergence = True
    print("energy : " +str(energy))
    # and 20 < np.abs(DIFF_CUT)
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


