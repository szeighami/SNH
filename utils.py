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

def get_train_test(data_loc, cell_width, max_val, augmented_query_size, utilization_range, test_size, dim, eps, test_query_loc, db, min_vals, max_vals, random_seed):
    #max_dist_thres = 0.32

    # creates a histogram of dataset
    t_queries_x = np.arange(start=-max_val/2-augmented_query_size, stop=max_val/2+augmented_query_size, step=cell_width)
    t_queries_y = np.arange(start=-max_val/2-augmented_query_size, stop=max_val/2+augmented_query_size, step=cell_width)
    total_cells = t_queries_x.shape[0]
    xx, yy =  np.meshgrid(t_queries_x, t_queries_y, indexing='ij')
    grid_center = np.dstack([xx, yy]).reshape(-1, dim)

    H, xedges, yedges = np.histogram2d(db[:, 0], db[:, 1], bins=[np.append(t_queries_x,t_queries_x[-1]+cell_width), np.append(t_queries_y,t_queries_y[-1]+cell_width)], range=[[-max_val/2, max_val/2], [-max_val/2, max_val/2]])
    grid_val = H.astype(int)


    grid_locs = grid_center.reshape(total_cells, total_cells, 2)
    total_in_cell = int((max_val+augmented_query_size)/cell_width)
    answer_len = np.floor(augmented_query_size/cell_width).astype(int)
    extra_len =  augmented_query_size - answer_len*cell_width
    qs_begin = grid_locs[:total_in_cell-answer_len, :total_in_cell-answer_len]
    qs_end = grid_locs[:total_in_cell-answer_len, :total_in_cell-answer_len]+augmented_query_size
    qs = (qs_begin+qs_end)/2
    qs = qs.reshape(-1, 2)

    grid_noise  = 1/eps
    #fixing the random seed so that noise value is the same across different neural networks
    np.random.seed(random_seed)
    grid_val_noisy =  grid_val+np.random.laplace(0, grid_noise, grid_val.shape)

    if answer_len == 0:
        extra_frac = (extra_len/cell_width)
        ress = np.array([[grid_val_noisy[i+answer_len, j+answer_len]*(extra_frac**2)] for i in range(total_in_cell-answer_len) for j in range(total_in_cell-answer_len)]).reshape(-1, 1)
    else:
        ress = np.array([[np.sum(grid_val_noisy[i:i+answer_len, j:j+answer_len])] for i in range(total_in_cell-answer_len) for j in range(total_in_cell-answer_len)]).reshape(-1, 1)
        if extra_len > 0:
            extra_frac = (extra_len/cell_width)
            extra_res = np.array([[(np.sum(grid_val_noisy[i:i+answer_len+1, j:j+answer_len+1])-np.sum(grid_val_noisy[i:i+answer_len, j:j+answer_len])-grid_val_noisy[i+answer_len, j+answer_len])*extra_frac + grid_val_noisy[i+answer_len, j+answer_len]*(extra_frac**2)] for i in range(total_in_cell-answer_len) for j in range(total_in_cell-answer_len)]).reshape(-1, 1)
            ress = ress+extra_res

    ress = np.clip(ress, 0, None) 

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
