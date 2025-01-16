import os
from mod_opt.data_load import *
from sklearn.neural_network import MLPRegressor

# Define hyperparameters and their possible values
hyperparameters = {
    'hidden_layer_sizes': [(100,), (25, 10), (25, 10, 5), (50, 25, 10), (50, 25, 10, 5), (100, 50, 25, 10), (100, 50, 25, 10, 5), (150, 100, 50, 25, 10, 5)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.005, 0.001, 0.05, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 500],
    'random_state': [42]
}

# Get a list of all files in the directory
files = os.listdir('feat_filter/filtered_features')

print('MLP Regressor model iteration started!')
for file in files:
    # Skip directories
    if os.path.isdir('feat_filter/filtered_features/' + file):
        continue

    # Read the selected features from the file
    selected_feat = [line.strip() for line in open('feat_filter/filtered_features/' + file, 'r')]
    feat = selected_feat

    # Perform the grid search and save the results
    result_mlp = grid_search_(MLPRegressor, X_train_norm[feat], y_train, X_val_norm[feat], y_val, hyperparameters)
    result_mlp.sort_values(by='r2_adj',ascending=False).to_csv('mod_opt/model_scores/mlp_scores_' + os.path.splitext(file)[0] + '.csv')

print('MLP Regressor model iteration completed!')