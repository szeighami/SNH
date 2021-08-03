import time
import json
import sys
import numpy as np
import pandas as pd
from jax_model import Phi 
import itertools
from functools import partial
import haiku as hk
import jax
from jax import value_and_grad, jit, random
from jax.experimental import optimizers
import jax.numpy as jnp
from utils import Log, MAE, SAE, mse_weighted_loss, calc_metrics,  get_train_test, get_training_weights



with open('conf.json') as f:
    config = json.load(f) 
    max_val = config['MAX_VAL']
    dim = config['in_dim']
    test_size = config['test_size']
    eps = config['eps']
    rho = config['rho']
    max_data_size = config['n']
    test_query_loc = config['test_query_loc']
    q_w_loc = config['q_w_loc']
    data_loc = config['data_loc']
    augmented_query_size = config['augmented_query_size']
    out_dim=config['out_dim']
    in_dim=config['in_dim']
    model_width=config['model_width']
    model_depth=config['model_depth']
    lr = config['lr']
    batch_size = config["training_batch_size"]
    utilization_range = 2*config["utilization_range"]
    epochs = config['EPOCHS']
    random_seed = config['random_seed']


db = np.load(data_loc)[:max_data_size]
n = db.shape[0]

min_vals = np.min(db, axis=0)
max_vals = np.max(db, axis=0)
db = ((db-min_vals)/(max_vals-min_vals)-0.5)*max_val


print("Creating model for query size " +str(100*config['augmented_query_size']/config['MAX_VAL'])+ "±"+ str(100*config['utilization_range']/config['MAX_VAL'])+" % of query space")
print("Preparing training data")
qs, ress, test, test_res = get_train_test(data_loc, rho, max_val, augmented_query_size, utilization_range, test_size, dim, eps, test_query_loc, db, min_vals, max_vals, random_seed)
print("Calculating training weights")
weights, counts = get_training_weights(q_w_loc, n, max_vals, min_vals, max_val, qs, ress, augmented_query_size)

qs = qs[counts.reshape(-1)>0, :]
ress = ress[counts.reshape(-1)>0, :]
weights = weights[counts.reshape(-1)>0, :]

print("initializing the model")
# 1. 'partial' fixes arguments of the function
# 2. hk.transform turns functions that use these object-oriented, functionaly "impure" modules into pure functions that can be used with jax.jit, jax.grad, jax.pmap, etc.
model = hk.transform(partial(Phi, out_dim=out_dim, in_dim=in_dim, width=model_width, no_layers=model_depth))


opt_init, opt_update, get_params = optimizers.adam(lr) # returns training handles
def train_fn(_, i, opt_state, batch):
    params = get_params(opt_state)
    loss_value, grads = value_and_grad(partial(loss, model, weights))(params, batch)
    return opt_update(i, grads, opt_state), loss_value

loss = mse_weighted_loss
metrics = [MAE(), SAE(n*0.001, "rel. error")]


key = random.PRNGKey(1)
init_params = model.init(key, qs) # nneed to specify shape to compiler
opt_state = opt_init(init_params)
itercount = itertools.count()


if batch_size > qs.shape[0]:
    batch_size = qs.shape[0]
no_batches = qs.shape[0]//batch_size


# jit or NOT
train_fn = jit(train_fn)
cum_duration = 0
logs = Log()

print("training")

for epoch in range(1, epochs + 1):
    start = time.perf_counter()
    cum_loss = 0

    #shuffle
    p = np.random.permutation(len(qs))
    qs = qs[p]
    ress = ress[p]
    weights = weights[p]

    for batch in range(no_batches):
        mega_batch = (qs[batch*batch_size:(batch+1)*batch_size], ress[batch*batch_size:(batch+1)*batch_size], weights[batch*batch_size:(batch+1)*batch_size])
        opt_state, loss_value = train_fn(
            key,
            next(itercount),
            opt_state,
            mega_batch
        )

    logs.add("loss", loss_value)
    calc_metrics(model, get_params(opt_state), (test, test_res), metrics, logs, augmented_query_size)

    duration = time.perf_counter() - start
    cum_duration += duration
    out_str = str(epoch)+" Loss: " + str(loss_value)+" "
    for metric in metrics:
        out_str += metric.name +": " +str(logs.get(metric.name)) +" "

    out_str += " time : " +str(cum_duration) +" "

    if epoch % 100 == 0:
        print(out_str)
        logs.save()

