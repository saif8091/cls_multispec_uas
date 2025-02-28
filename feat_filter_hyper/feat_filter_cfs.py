import os
import numpy as np
import pandas as pd
from src.feature_selection import FilterBasedFeatureSelection

dir_name = 'feat_filter_hyper/filtered_features'

# Check if the directory exists
if not os.path.exists(dir_name):
    # If the directory doesn't exist, create it
    os.makedirs(dir_name)

print('Performing correlation based feature selection')
with pd.HDFStore('data/preprocessed_data/feat_h_split_comb.h5', mode='r') as store:
    X_train = store['X_train']
    y_train = store['y_train']
    X_val = store['X_val']
    y_val = store['y_val']
    X_test = store['X_test']
    y_test = store['y_test']

    labels_train = store['labels_train']
    labels_val = store['labels_val']
    labels_test = store['labels_test']

cfs = FilterBasedFeatureSelection(pd.concat([X_train,X_val]),pd.concat([y_train,y_val]),handle='cor')
selected_features = cfs.select_based_on_threshold(min_r2=0.30, r2_thresh=0.80)
open('feat_filter_hyper/filtered_features/cfs.txt', 'w').writelines(f"{item}\n" for item in selected_features)
print(f'Number of features selected for cfs: {len(selected_features)}')