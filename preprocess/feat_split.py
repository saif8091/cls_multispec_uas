'''
This script is used to split the data into train, validation and test sets.
'''
import numpy as np
import pandas as pd
import numpy as np
from src.misc import remove_columns_ending_with_0
from sklearn.model_selection import train_test_split

#################### Input parameters ##################################

#### Specify the file path where you want to save the data
file_path = 'data/preprocessed_data/feat_split_comb.h5'
#### Test set split and val split
test_set_split = 0.3
val_set_split = 0.2     # note this split ratio is after the first ratio was applied

#### data to split
Data_21_22=pd.read_csv('data/preprocessed_data/feat_21_22.csv', header=0)
Data23 = pd.read_csv('data/preprocessed_data/feat_23.csv', header=0)
########################################################################

##### combining 21 and 23 data
Data_21_22 = remove_columns_ending_with_0(Data_21_22)   # removes blue features
Data21 = Data_21_22.iloc[:200]
Data22 = Data_21_22.iloc[200:]
Data = pd.concat([Data21, Data22, Data23])

labels = Data[['Flight', 'Plot']]
X = Data.iloc[:,2:-1]
y = Data['CLS_score']

X_train_val, X_test, y_train_val, y_test, labels_train_val, labels_test = train_test_split(X, y, labels, test_size=test_set_split, random_state=42)
X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(X_train_val, y_train_val, labels_train_val, test_size=val_set_split, random_state=42)

# Create a dictionary to store your dataframes
dataframes = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,
    'labels_train': labels_train,
    'labels_val': labels_val,
    'labels_test': labels_test
}

# Save the dataframes to an HDF5 file
with pd.HDFStore(file_path, mode='w') as store:
    for key, df in dataframes.items():
        store.put(key, df)

print('Train shape: ', X_train.shape)
print('Validation shape: ', X_val.shape)
print('Test shape: ', X_test.shape)
print('Data processing done, the file can be found in ',file_path)