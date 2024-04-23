import numpy as np
import pandas as pd
from src.feature_selection import FilterBasedFeatureSelection

print('Performing mutual information based feature selection')
with pd.HDFStore('data/preprocessed_data/feat_split_comb.h5', mode='r') as store:
    X_train = store['X_train']
    y_train = store['y_train']
    X_val = store['X_val']
    y_val = store['y_val']
    X_test = store['X_test']
    y_test = store['y_test']

    labels_train = store['labels_train']
    labels_val = store['labels_val']
    labels_test = store['labels_test']

mfs = FilterBasedFeatureSelection(pd.concat([X_train,X_val]),pd.concat([y_train,y_val]),handle='mi')
selected_features = mfs.select_based_on_threshold(min_r2=0.50, r2_thresh=0.90)
open('feat_filter/filtered_features/mfs.txt', 'w').writelines(f"{item}\n" for item in selected_features)
print(f'Number of features selected for mfs: {len(selected_features)}')