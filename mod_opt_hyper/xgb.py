import os
from mod_opt_hyper.data_load import *
import xgboost as xgb

# Define hyperparameters and their possible values
hyperparameters = {
    'n_estimators': [50, 100, 500, 1000], # I'm noticing similar performance for 1000 and 1500
    'learning_rate': [0.05, 0.07, 0.1, 0.15, 0.2], # increased learning rate doesn't yield good result stick with below 0.07
    'max_depth': [1, 2, 3, 4, 7],
    'objective': ['reg:squarederror']
}
def xgb_feat_imp(model,X,y):
    if not isinstance(model, xgb.XGBRegressor):
        raise ValueError("Model must be an XGBoost Regressor")
    return model.feature_importances_

# Get a list of all files in the directory
files = os.listdir('feat_filter_hyper/filtered_features')

print('Xtreme gradient boost iteration started!')
for file in files:
    # Skip directories
    if os.path.isdir('feat_filter_hyper/filtered_features/' + file):
        continue

    # Read the selected features from the file
    selected_feat = [line.strip() for line in open('feat_filter_hyper/filtered_features/' + file, 'r')]
    feat = selected_feat

    # Perform the grid search and save the results
    result_xgb = grid_search_(xgb.XGBRegressor, X_train_norm[feat], y_train, X_val_norm[feat], y_val, hyperparameters,feature_importance_func=xgb_feat_imp)
    result_xgb.sort_values(by='r2_adj',ascending=False).to_csv('mod_opt_hyper/model_scores/xgb_scores_' + os.path.splitext(file)[0] + '.csv')

print('Xtreme gradient boost iteration completed!')