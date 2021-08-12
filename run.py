import numpy as np
import os
import pandas as pd
import subprocess
import json
import gzip
import pickle

from utils import get_auxiliary_features

curr_dir = os.path.dirname(os.path.realpath(__file__))
config={}

config['NAME'] = 'test_sf_cabs'  # Experiment name, the output will be written in tests/config['NAME']

config['data_loc'] = curr_dir+"/data/CABS_SFS.npy"  # path to private dataset
config['n'] = 217438  # private dataset size
config['eps'] = 0.2  # privacy budget

config['q_w_loc'] = curr_dir+"/data/gowalla_SF.npy"# path to auiliary public dataset, q_w

#testing information
config['test_query_loc'] = config['data_loc']  # path to test queries, same as data if we consider query locations that follow data distribution (only accessed at test time), query size is uniform between min_test_query_size and max_test_query_size
config["min_test_query_size"] = 0.02  # minimum test quer size, specified so that queries are at least of size 100*config["min_test_query_size"]/config['MAX_VAL'] percent of query space
config["max_test_query_size"] = 0.08  # maximum test quer size, specified so that queries are at most of size 100*config["max_test_query_size"]/config['MAX_VAL'] percent of query space
config['MAX_VAL'] = 10  # normalization constant, so that datapoins are normalized to be between -config['MAX_VAL']/2 and  +config['MAX_VAL']/2
config['test_size'] = 10000  # number of test queries to ask

# system parameters
config['no_models'] = 8
config['rho']=0.01544

use_paramselect = True
if use_paramselect:
    with gzip.open('paramselect/ParamSelect_trained_model.pklz', 'r') as f:
        est1 = pickle.load(f)
    feature_list = [[config['n'], 
                    config['eps'], 
                    1/(config['n']*config['eps']), 
                    np.sqrt(1/(config['n']*config['eps'])), 
                    get_auxiliary_features(config['q_w_loc']),
                   ]]
    feature_names = ['n','eps','neps','sqrtneps', 'entropy_512']
    X = pd.DataFrame(feature_list, columns=feature_names)
    pred_rho = est1.predict(X)
    print('ParamSelect predicted rho: ', pred_rho)
    config['rho']  = pred_rho[0]

# model hyper-parameters
config['lr']=0.001
config['training_batch_size']=150000
config['EPOCHS'] = 10000
config['model_width']=80
config['model_depth']=20
config['out_dim'] = 1
config['in_dim'] = 2 
config['random_seed'] = np.random.randint(1)

config["py_loc"] = os.path.dirname(os.path.realpath(__file__))+'/'

# finding ranges to train models at
begin_range = config["min_test_query_size"]
end_range = config["max_test_query_size"]
config['utilization_range'] = (end_range-begin_range)/config['no_models']*0.5
R = [(end_range-begin_range)*(j+0.5)/config['no_models'] + begin_range for j in range(config['no_models'])]
for i, augmented_query_size in enumerate(R):
    config['augmented_query_size']=augmented_query_size

    os.system('mkdir tests/')
    os.system('mkdir tests/'+ config['NAME'])
    os.system('mkdir tests/'+ config['NAME']+'/'+str(i))
    with open('tests/'+config['NAME']+'/'+str(i)+'/conf.json', 'w') as f:
        json.dump(config, f)

    command = 'cd tests/'+ config['NAME'] +'/'+str(i) + ' &&  XLA_PYTHON_CLIENT_PREALLOCATE=false python -u '+config["py_loc"]+'fit_base_JAX.py > out.txt '
    os.system(command)
