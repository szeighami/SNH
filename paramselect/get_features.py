import pandas as pd
import numpy as np
from scipy.stats import entropy

def get_auxiliary_features(filename): 
    for _bins in [512]:
        df = pd.read_csv(filename)
        H, xedges, yedges = np.histogram2d(df.iloc[:, 0].values, df.iloc[:, 1].values, bins=_bins)
        x_probs = np.true_divide(H,np.sum(H)) # convert the histogram to probability
        x_probs = x_probs.ravel() # flatten
        ent = entropy(x_probs) # shannon entropy
        print('Determined feature for ',filename, '\t at binning:', _bins, ' size: ' np.sum(H),'\t entropy: ', ent)
        return ent