import os
from mod_opt_hyper.data_load import *
from sklearn.cross_decomposition import PLSRegression

def reg_coef(model,X,y):
    return np.abs(model.coef_)

# Get a list of all files in the directory
files = os.listdir('feat_filter_hyper/filtered_features')

print('Partial least square regression iteration started!')
for file in files:
    # Skip directories
    if os.path.isdir('feat_filter_hyper/filtered_features/' + file):
        continue
    
    # Read the selected features from the file
    selected_feat = [line.strip() for line in open('feat_filter_hyper/filtered_features/' + file, 'r')]
    feat = selected_feat
    hyperparameters = {'n_components': list(range(1,len(feat)+1))}
    # Perform the grid search and save the results
    result_plsr = grid_search_(PLSRegression,X_train_norm[feat],y_train,X_val_norm[feat],y_val,hyperparameters,reg_coef)
    result_plsr.sort_values(by='r2_adj',ascending=False).to_csv('mod_opt_hyper/model_scores/plsr_scores_' + os.path.splitext(file)[0] + '.csv')

print('Partial least square regression iteration completed!')