import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from src.misc import *
from src.model_search import *
from src.model_utils import *

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

X_train_norm,X_val_norm, norm_model = normalize_features(X_train,X_val)
X_train_norm,X_test_norm,_ = normalize_features(X_train,X_test)