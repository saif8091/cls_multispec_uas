import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.utils import interpolate_dataframe
from src.dates import *

# Define the directory
dir_name = 'data/preprocessed_data'

# Check if the directory exists
if not os.path.exists(dir_name):
    # If the directory doesn't exist, create it
    os.makedirs(dir_name)

# Load the data
excel = pd.read_csv('data/CLS_DS.csv')

# Extract the data for 2021, 2022, and 2023
cls_2021 = excel[['Plot','D0','D1','D2', 'D3','D4','D5']][:40]
cls_2021.set_index('Plot', inplace=True)
cls_2021.columns = d_day_2021

cls_2022 = excel[['Plot','D0','D1','D2', 'D3','D4','D5']][40:80]
cls_2022.set_index('Plot', inplace=True)
cls_2022.columns = d_day_2022

cls_2023 = excel[['Plot','D0','D1','D2', 'D3','D4','D5','D6']][80:]
cls_2023.set_index('Plot', inplace=True)
cls_2023.columns = d_day_2023

# Interpolate the data
cls_interp_2021 = interpolate_dataframe(cls_2021, f_day_2021, method='quadratic')
cls_interp_2022 = interpolate_dataframe(cls_2022, f_day_2022, method='quadratic')
cls_interp_2023 = interpolate_dataframe(cls_2023, f_day_2023, method='quadratic')

# modify the column names to flight dates
cls_interp_2021.columns = ['20210707','20210715', '20210720', '20210802', '20210825']
cls_interp_2022.columns = ['20220707','20220715', '20220726', '20220810', '20220818']
cls_interp_2023.columns = ['20230802','20230821', '20230828', '20230906', '20230911']

# Save the interpolated data
pd.concat([cls_interp_2021, cls_interp_2022],axis=1).to_csv(f'{dir_name}/cls_interp_2021_2022.csv')
cls_interp_2023.to_csv(f'{dir_name}/cls_interp_2023.csv')

print(f'CLS interpolation complete! Files can be found in the {dir_name} directory.')