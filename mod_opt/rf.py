import os
from mod_opt.data_load import *
from sklearn.ensemble import RandomForestRegressor

# Define hyperparameters and their possible values
hyperparameters = {
    'n_estimators': [25, 50, 100, 200, 500, 700, 1000, 1200],
    'max_depth': [3, 5, 8, 12, 15, 18],
    # Fixed hyperparameter
    'min_samples_split': [3],
    'min_samples_leaf': [2],
    'max_features': ['log2'],
    'random_state': [42]
}
def r_f_feat_imp(model,X,y):
    if not isinstance(model, RandomForestRegressor):
        raise ValueError("Model must be a RandomForestRegressor instance")

    return model.feature_importances_

# Get a list of all files in the directory
files = os.listdir('feat_filter/filtered_features')

print('Random Forest model iteration started!')
for file in files:
    # Skip directories
    if os.path.isdir('feat_filter/filtered_features/' + file):
        continue

    # Read the selected features from the file
    selected_feat = [line.strip() for line in open('feat_filter/filtered_features/' + file, 'r')]
    feat = selected_feat

    # Perform the grid search and save the results
    result_rf = grid_search_(RandomForestRegressor, X_train_norm[feat], y_train, X_val_norm[feat], y_val, hyperparameters,feature_importance_func=r_f_feat_imp)
    result_rf.sort_values(by='r2_adj',ascending=False).to_csv('mod_opt/model_scores/rf_scores_' + os.path.splitext(file)[0] + '.csv')

print('Random Forest model iteration completed!')