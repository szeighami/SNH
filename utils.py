import json
import numpy as np
import pandas as pd
import jax
from jax import value_and_grad, grad, jit, random, vmap
from jax.experimental import optimizers, stax
from jax.lib import pytree
from jax.tree_util import tree_flatten, tree_multimap, tree_unflatten
import jax.numpy as jnp
from rtree import index
from scipy.stats import entropy
from scipy import sparse
import time


print_for_debug = False

def get_auxiliary_features(filename): 
        _bins = 512
        aux_data = np.load(filename)
        H, xedges, yedges = np.histogram2d(aux_data[:, 0], aux_data[:, 1], bins=_bins)
        x_probs = np.true_divide(H,np.sum(H)) # convert the histogram to probability
        x_probs = x_probs.ravel() # flatten
        ent = entropy(x_probs) # shannon entropy
#         print('Determined feature for dataset ',filename, ' of size ', np.sum(H), ' at binning ', _bins, ' entropy: ', ent)
        return ent

class Log():
    def __init__(self):
        self.log = {}

    def add(self, name, val):
        if name not in self.log:
            self.log[name] = []
        self.log[name].append(float(val))

    def get(self, name):
        return self.log[name][-1]

    def save(self, path='results.json'):
        log_df = pd.DataFrame.from_dict(self.log)
        with open(path, 'w') as f:
            log_df.to_json(f)


class MAE():
    def __init__(self):
        self.name = "mae"

    def call(self, model, params, batch, weights=None):
        inputs, y_true = batch[0], batch[1]
        y_pred = model.apply(params, None, inputs)
        return jnp.average(jnp.abs((y_pred - y_true)**2), axis=0)

    def calc(self, y_true, y_pred):
        return jnp.average(jnp.abs(y_pred - y_true), axis=0)

class SAE():
    def __init__(self, smooth, name):
        self.name = name
        self.smooth = smooth

    def call(self, model, params, batch, weights=None):
        inputs, y_true = batch[0], batch[1]
        y_pred = model.apply(params, None, inputs)
        return jnp.average(jnp.abs(y_pred - y_true)/(jnp.maximum(y_true, self.smooth)), axis=0)

    def calc(self, y_true, y_pred):
        return jnp.average(jnp.abs(y_pred - y_true)/(jnp.maximum(y_true, self.smooth)), axis=0)

def mse_weighted_loss(model, weights, params, batch):
    inputs, y_true, _weights = batch[0], batch[1], batch[2]
    y_pred = model.apply(params, None, inputs)
    return jnp.average(jnp.square(jnp.subtract(y_pred, y_true)), weights=_weights)

def calc_metrics(model, params, batch, metrics, logs, model_query_size,  weights=None):
    inputs, y_true = batch[0], batch[1]
    if inputs.shape[1] == 3:
        x = inputs[:, :2]
    else:
        x = inputs
    y_pred = model.apply(params, None, x)

    if inputs.shape[1] == 3:
        y_pred = y_pred*((inputs[:, 2]/model_query_size)**2)


    for metric in metrics:
        val = metric.calc(y_true, y_pred) # nonunifrom test set
        logs.add(metric.name, val[0])

# TODO::: write new get_train_test_categorical function foir ADULT dataset. 

def get_train_test(data_loc, cell_width, max_val, augmented_query_size, utilization_range, test_size, dim, eps, test_query_loc, db, min_vals, max_vals, random_seed, get_cell_aligned_qs = True, k = 3):
    #max_dist_thres = 0.32

    # creates a histogram of dataset

    # push the extents a little larger based on augmented_query_size
    t_queries_x = np.arange(start=-max_val/2-augmented_query_size, stop=max_val/2+augmented_query_size, step=cell_width)
    t_queries_y = np.arange(start=-max_val/2-augmented_query_size, stop=max_val/2+augmented_query_size, step=cell_width)
    total_cells = t_queries_x.shape[0]
    xx, yy =  np.meshgrid(t_queries_x, t_queries_y, indexing='ij')
    grid_corners = np.dstack([xx, yy]).reshape(-1, dim)

    H, xedges, yedges = np.histogram2d(db[:, 0], db[:, 1], bins=[np.append(t_queries_x,t_queries_x[-1]+cell_width), np.append(t_queries_y,t_queries_y[-1]+cell_width)], range=[[-max_val/2, max_val/2], [-max_val/2, max_val/2]])

    grid_val = H.astype(int)
    grid_noise  = 1/eps
    #fixing the random seed so that noise value is the same across different neural networks
    np.random.seed(random_seed)
    grid_val_noisy =  grid_val+np.random.laplace(0, grid_noise, grid_val.shape)

    grid_locs = grid_corners.reshape(total_cells, total_cells, 2)
    total_in_cell = int((max_val+augmented_query_size)/cell_width)
    answer_len = np.floor(augmented_query_size/cell_width).astype(int)
    extra_len =  augmented_query_size - answer_len*cell_width
    qs_begin = grid_locs[:total_in_cell-answer_len, :total_in_cell-answer_len]
    qs_end = grid_locs[:total_in_cell-answer_len, :total_in_cell-answer_len]+augmented_query_size
    qs = (qs_begin+qs_end)/2
    qs = qs.reshape(-1, 2)
    
    if print_for_debug:
        print('grid_val_noisy', grid_val_noisy[0:2,0:2], grid_val_noisy.shape)
        print('xx', xx[0:2,0:2], xx.shape)
        print('yy', yy[0:2,0:2], yy.shape)
        print('total_cells', total_cells)
        print('total_in_cell', total_in_cell)
        print('augmented_query_size', augmented_query_size)
        print('cell_width', cell_width)
        print('qs', qs[0],qs.shape)
        print('answer_len', answer_len)
        print('extra_len', extra_len)

    if get_cell_aligned_qs:
        if answer_len == 0:
            extra_frac = (extra_len/cell_width)
            ress = np.array([[grid_val_noisy[i+answer_len, j+answer_len]*(extra_frac**2)] for i in range(total_in_cell-answer_len) for j in range(total_in_cell-answer_len)]).reshape(-1, 1)
        else:
            ress = np.array([[np.sum(grid_val_noisy[i:i+answer_len, j:j+answer_len])] for i in range(total_in_cell-answer_len) for j in range(total_in_cell-answer_len)]).reshape(-1, 1)
            if extra_len > 0:
                extra_frac = (extra_len/cell_width)
                extra_res = np.array([[(np.sum(grid_val_noisy[i:i+answer_len+1, j:j+answer_len+1])-np.sum(grid_val_noisy[i:i+answer_len, j:j+answer_len])-grid_val_noisy[i+answer_len, j+answer_len])*extra_frac + grid_val_noisy[i+answer_len, j+answer_len]*(extra_frac**2)] for i in range(total_in_cell-answer_len) for j in range(total_in_cell-answer_len)]).reshape(-1, 1)
                ress = ress+extra_res
        if print_for_debug:
            print('ress', ress[0:5], ress.shape)
        ress = np.clip(ress, 0, None)

    else:
        # Q_new = [Q+i*(c/k) for i in range(k)], so basically when k=1 you get the same answers as the block above
        assert k >= 1, 'what do you mean k is zero'
        start = time.time()
        qsess = []
        for i in range(k):
            _qs = qs + (i * (cell_width/k))
            qsess.append(_qs)
        qs = np.concatenate(qsess)
        ress = get_ress_from_grid(qs, augmented_query_size/2, cell_width, max_val, grid_val_noisy)
        ress = ress.reshape(-1, 1)
        print('Generated', len(ress), 'training queries in ' , time.time()-start, 'seconds')

    if print_for_debug:
        print('ress', ress[0:5], ress.shape)

    utilization_range_begin = augmented_query_size - utilization_range/2
    test_range = np.random.rand(test_size, 1)*(utilization_range)+utilization_range_begin
    test_loc = np.load(test_query_loc)
    test_loc = ((test_loc-min_vals)/(max_vals-min_vals)-0.5)*max_val
    np.random.shuffle(test_loc)
    test_loc = test_loc[:test_size]
    test = np.concatenate([test_loc, test_range], axis=1)
    test_res = np.array([np.sum([np.logical_and.reduce([np.logical_and((db[:, d]<test[i, d]+test[i, -1]/2),(db[:, d]>=test[i, d]-test[i, -1]/2)) for d in range(dim)])]) for i in range(test.shape[0])]).reshape((-1, 1))

    return qs, ress, test, test_res

def get_training_weights(q_w_loc, n, max_vals, min_vals, max_val, qs, ress, augmented_query_size):
    q_w = np.load(q_w_loc)
    q_w = ((q_w-min_vals)/(max_vals-min_vals)-0.5)*max_val


    idx = index.Index() # rtree index intialization
    for i in range(q_w.shape[0]): # insert all workload queries as points into the rtree
        idx.insert(i, (q_w[i][0],q_w[i][1]))

    counts = []
    for i in range(qs.shape[0]): # count for each query how many workload points are in it
        lb = qs[i] - augmented_query_size/2
        ur = qs[i] + augmented_query_size/2
        counts.append( idx.count((lb[0],lb[1],ur[0],ur[1])) )


    counts = np.array(counts)
    counts = counts.reshape((len(counts),1))


    weights = counts/(np.maximum(ress, 0.001*n))
    return weights, counts


# q_range is half auigemented query size
# x is H, the noisy counts in each cell
# test_queries are actually all centers of queries.
def get_ress_from_grid(test_queries, q_range, cell_width, max_val, x):

    test_queries_lb = test_queries - q_range
    test_queries_ru = test_queries + q_range
    test_queries_lu = np.concatenate([test_queries_lb[:, 0:1], test_queries_ru[:, 1:2]], axis=1)
    test_queries_rb = np.concatenate([test_queries_ru[:, 0:1], test_queries_lb[:, 1:2]], axis=1)
    
    if print_for_debug:
        print("test_queries_lb[0] before clip: ", test_queries_lb[0])
        print("test_queries_ru[0] before clip: ", test_queries_ru[0])
        print("test_queries_lu[0] ",test_queries_lu[0])
        print("test_queries_rb[0] ",test_queries_rb[0])

    max_val = max_val + (q_range*2*2) # umm yeah just extend augemented query size on both sides

    cell_area = (cell_width)**2
    test_indx_lb = np.clip(np.floor((test_queries_lb+max_val/2)/(cell_width)), 0, x.shape[0]-1).astype(int)
    test_indx_ru = np.clip(np.floor((test_queries_ru+max_val/2)/(cell_width)), 0, x.shape[0]-1).astype(int)
    test_indx_lu = np.clip(np.floor(((test_queries_lu+max_val/2))/(cell_width)), 0, x.shape[0]-1).astype(int)
    test_indx_rb = np.clip(np.floor(((test_queries_rb+max_val/2))/(cell_width)), 0, x.shape[0]-1).astype(int)

    if print_for_debug:
        print("test_indx_lb[0] ", ((test_queries_lb[0]+max_val/2))/(cell_width), test_indx_lb[0])
        print("test_indx_ru[0] ", ((test_queries_ru[0]+max_val/2))/(cell_width), test_indx_ru[0])
        print("test_indx_lu[0] ", ((test_queries_lu[0]+max_val/2))/(cell_width), test_indx_lu[0])
        print("test_indx_rb[0] ", ((test_queries_rb[0]+max_val/2))/(cell_width), test_indx_rb[0])

    qs = []
    # qs_sparse = []
    # set num of queries to evaluate

    # num_qs = 10000
    num_qs = len(test_queries)

    start = time.time()

    ress_np = np.zeros(num_qs, dtype=np.float32)
    # last_end = -1
    # batch_size = 5000
    # 10k queries batch size 100 = 14 seeconds
    # 10k queries batch size 500 = 14 seeconds
    # 10k queries batch size 1000 = 14 seeconds
    # batch size makes almost no difference in terms of running time. just set it to something that can fit in memory
    # 5k queries take 20 % memory for 642*642 grid.

    # x_sp = sparse.csr_matrix(x)
    for i in range(num_qs):
        q = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
        # q = sparse.csr_matrix((x.shape[0], x.shape[1]), dtype=np.float32)
     
        lb_intersection = np.floor(((test_queries_lb[i]+max_val/2)/(cell_width)))*cell_width+np.array([cell_width, cell_width])-max_val/2
        rb_intersection = np.floor(((test_queries_rb[i]+max_val/2)/(cell_width)))*cell_width+np.array([0, cell_width])-max_val/2
        lu_intersection = np.floor(((test_queries_lu[i]+max_val/2)/(cell_width)))*cell_width+np.array([cell_width, 0])-max_val/2
        ru_intersection = np.floor(((test_queries_ru[i]+max_val/2)/(cell_width)))*cell_width+np.array([0, 0])-max_val/2

        if i == 0 and print_for_debug:
            print("lb_intersection ", lb_intersection)
            print("rb_intersection ", rb_intersection)
            print("lu_intersection ", lu_intersection)
            print("ru_intersection ", ru_intersection)

        area_lb = np.prod(lb_intersection-test_queries_lb[i])
        area_rb = np.prod(np.abs(rb_intersection-test_queries_rb[i]))
        area_lu = np.prod(np.abs(lu_intersection-test_queries_lu[i]))
        area_ru = np.prod(np.abs(ru_intersection-test_queries_ru[i]))

        if i == 0 and print_for_debug:
            print("area_lb ", area_lb, area_lb/cell_area)
            print("area_rb ", area_rb, area_rb/cell_area)
            print("area_lu ", area_lu, area_lu/cell_area)
            print("area_ru ", area_ru, area_ru/cell_area)

        q[test_indx_lb[i, 0], test_indx_lb[i, 1]] = area_lb/cell_area
        q[test_indx_rb[i, 0], test_indx_rb[i, 1]] = area_rb/cell_area
        q[test_indx_lu[i, 0], test_indx_lu[i, 1]] = area_lu/cell_area
        q[test_indx_ru[i, 0], test_indx_ru[i, 1]] = area_ru/cell_area
        q[test_indx_lb[i, 0]+1:test_indx_rb[i, 0], test_indx_lb[i, 1]+1:test_indx_lu[i, 1]]=1

        if test_indx_rb[i, 0] - test_indx_lb[i, 0]>1:
            portion_mid = (lb_intersection[1]-test_queries_lb[i,1])/(cell_width)
            q[test_indx_lb[i, 0]+1:test_indx_rb[i, 0], test_indx_rb[i, 1]] = portion_mid
        if test_indx_ru[i, 1] - test_indx_rb[i, 1]>1:
            portion_mid = (test_queries_rb[i,0] - rb_intersection[0])/(cell_width)
            q[test_indx_rb[i, 0], test_indx_rb[i, 1]+1:test_indx_ru[i, 1]] = portion_mid
        if test_indx_ru[i, 0] - test_indx_lu[i, 0]>1:
            portion_mid = np.abs(lu_intersection[1]-test_queries_lu[i,1])/(cell_width)
            q[test_indx_lu[i, 0]+1:test_indx_ru[i, 0], test_indx_ru[i, 1]] = portion_mid
        if test_indx_lu[i, 1] - test_indx_lb[i, 1]>1:
            portion_mid = np.abs(test_queries_lb[i,0] - lb_intersection[0])/(cell_width)
            q[test_indx_lb[i, 0], test_indx_lb[i, 1]+1:test_indx_lu[i, 1]] = portion_mid

        # this is slow as shit but memory efficient
        # ress_np[i]= (q).multiply(x_sp).sum() 

        ress_np[i]= np.sum((q*x) )

        if i == 0 and print_for_debug:
            rows, cols = np.nonzero(q)
            print("non empty vals len(q[rows, cols])", len(q[rows, cols]))
            print(q[np.nonzero(q)])

    if print_for_debug:
        print('ress_np', ress_np[0:5], len(ress_np))
        print('time taken to generate', num_qs, 'training queries' , time.time()-start, 'seconds')

    # dt = qs_sparse.multiply(sparse.csr_matrix(x))
    # ress_from_sparse = sparse.sum(dt, axis=(1, 2))
    # print('ress_from_sparse', ress_from_sparse)
    return ress_np
