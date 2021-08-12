
# feature_generation notebook 
Notebook discusses how to generate features for model training.
File city_features_fromGW.csv saves the features once generated.

# model_traninig notebook
This notebook trains the ParamSelect model on the training data 
from file best_rho_GW_train_test.csv. After training, model is saved as ParamSelect_trained_model.pklz
, and can then be utilized from run.py to train SNH at predicted values of grid width.
