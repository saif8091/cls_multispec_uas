import os
from mod_opt.data_load import *
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

# Define hyperparameters and their possible values
hyperparameters = {
    'kernel': ['rbf','poly','linear', 'sigmoid'],
    'C': [1, 10, 100],
    'epsilon': [0.0001, 0.001, 0.01, 0.1],
    'gamma': ['scale'],
}

def svr_permutation_importance(model, X_train, y_train):
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=0)
    return result.importances_mean

# Get a list of all files in the directory
files = os.listdir('feat_filter/filtered_features')

print('Support vector regression iteration started!')
for file in files:
    # Skip directories
    if os.path.isdir('feat_filter/filtered_features/' + file):
        continue
    
    # Read the selected features from the file
    selected_feat = [line.strip() for line in open('feat_filter/filtered_features/' + file, 'r')]
    feat = selected_feat

    # Perform the grid search and save the results
    result_svr = grid_search_(SVR,X_train_norm[feat],y_train,X_val_norm[feat],y_val,hyperparameters,feature_importance_func=svr_permutation_importance)
    result_svr.sort_values(by='r2_adj',ascending=False).to_csv('mod_opt/model_scores/svr_scores_' + os.path.splitext(file)[0] + '.csv')

print('Support vector regression iteration completed!')