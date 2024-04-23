import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from itertools import product
import time
from tqdm import tqdm

def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)

    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    adjusted_r2_val = 1 - ((1 - r2_val) * (n_train - 1) / (n_train - n_features - 1))

    return r2_train, rmse_train, r2_val, rmse_val, adjusted_r2_val, model


def grid_search_(regressor, X_train, y_train, X_val, y_val, hyperparameters, 
                 feature_importance_func=None, min_features=1):
    ''' 
    This function generates r2 and rmse values for various hyperparameters
    Additionally, if feature importance function is specified it also performs
    recursive feature elimination for each hyperparameter setting

    Returns:
        A dataframe containing hyperparameter settings of the model and the name 
        of the features used along with its respective performances scores

    Example usage:
    # Define a dictionary of hyperparameters to search over
    hyperparameters = {
        'n_estimators': [50, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3],
        'objective' : ['reg:squarederror']
    }
    def xgboost_feature_importance(model):
        if not isinstance(model, xgb.XGBRegressor):
            raise ValueError("Model must be an XGBoost Regressor")

        return model.feature_importances_

    results = grid_search_(xgb.XGBRegressor, X_train, y_train, X_val, y_val,
                           hyperparameters, feature_importance_func= xgboost_feature_importance,
                           min_features=8)
    '''
    result_df = pd.DataFrame(columns=list(hyperparameters.keys()) + ['num_features', 'selected_features'] + 
                             ['r2_train', 'rmse_train', 'r2_val', 'rmse_val', 'r2_adj'])

    hyperparameter_combinations = list(product(*hyperparameters.values()))
    pbar = tqdm(total=len(hyperparameter_combinations), desc='Processing')
    
    for params in hyperparameter_combinations:
        hyperparams = dict(zip(hyperparameters.keys(), params))
        
        model = regressor(**hyperparams)
        selected_features = list(X_train.columns)  # Start with all features
        num_features = len(selected_features)
        
        if feature_importance_func:
            while num_features >= min_features:
                X_train_subset = X_train[selected_features]
                X_val_subset = X_val[selected_features]

                r2_train, rmse_train, r2_val, rmse_val, adjusted_r2_val, model = evaluate_model(model, X_train_subset, y_train, X_val_subset, y_val)

                # Append the results along with the number of features and the selected feature names
                result_df = result_df.append({**hyperparams, 'num_features': num_features, 'selected_features': selected_features.copy(),
                                             'r2_train': r2_train, 'rmse_train': rmse_train, 'r2_val': r2_val, 'rmse_val': rmse_val, 'r2_adj': adjusted_r2_val}, ignore_index=True)

                if (num_features <= min_features) or (r2_train < 0.7):     # If ensures that the code will ignore the hyperparameter combination that leads to r2 train lower than 0.8
                    break
                
                # Calculate feature importance scores
                feature_importance = feature_importance_func(model,X_train_subset,y_train)
                least_important_feature = sorted(zip(selected_features, feature_importance), key=lambda x: x[1])[0][0]
                selected_features.remove(least_important_feature)
                num_features -= 1
                
        else:
            r2_train, rmse_train, r2_val, rmse_val, adjusted_r2_val,_ = evaluate_model(model, X_train, y_train, X_val, y_val)

            result_df = result_df.append({**hyperparams, 'num_features': num_features, 'selected_features': selected_features,
                                         'r2_train': r2_train, 'rmse_train': rmse_train, 'r2_val': r2_val, 'rmse_val': rmse_val, 'r2_adj': adjusted_r2_val}, ignore_index=True)
        
        pbar.update(1)
    
    pbar.close()
    return result_df