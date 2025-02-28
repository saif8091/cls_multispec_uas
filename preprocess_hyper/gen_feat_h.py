"""
This module is used for generating features from preprocessed hyperspectral images and corresponding CLS scores.

It starts by importing necessary modules and loading preprocessed image data and CLS scores from CSV and pickle files. 

The module defines specific wavelength combinations for different years.

The main function in this module, `zip_feat_cls`, takes a wavelength combination,
a zipped vegetation image dictionary, and a CLS score dataframe as inputs. 
The function is used to generate features from the images and corresponding CLS scores.
"""
import os
import numpy as np
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from src.misc import *
from src.utils import *
from preprocess_hyper.prep_func import *
from src.feature_extraction import *

#### Preprocessed data directory #######
prep_dir = 'data/preprocessed_data/'

#### loading CLS score data
cls_score_21_22 =pd.read_csv(prep_dir + 'cls_interp_2021_2022.csv', header=0, index_col=0)

#### loading vegetation masked images
veg_h = pickle.load(open(prep_dir + 'veg_h.pkl', 'rb'))

#Load wavelength bands
lb, ub = 0, 80
hyper_wave = np.linspace(398.573,1001.81,272)
hyper_wave_3 = dn_sample_sig(hyper_wave, 3)[lb:ub]

def zip_feat_cls(wave_comb, zipped_veg_im, cls_score):
    '''
    This function takes a dictionary of images and a dataframe of cls score
    Returns a dataframe containing the features and cls score
    '''
    feat=[]
    pbarflt = tqdm(total=len(zipped_veg_im), desc=f'Total flights completed')
    for flight_date, flt_im in zipped_veg_im.items():
        pbarplt = tqdm(total=len(flt_im), desc=f'Extracting features for {flight_date}')
        for plot_num, veg_im in flt_im.items():
            feat.append({
            'Flight': flight_date,
            'Plot': plot_num,
            **features_from_single_veg_image(wave_comb,veg_im),
            'CLS_score': cls_score[flight_date][plot_num]/100  
            })
            pbarplt.update(1)
        pbarplt.close()
        pbarflt.update(1)
    pbarflt.close()
    return pd.DataFrame(feat)

feat_21_22 = zip_feat_cls(hyper_wave_3, veg_h, cls_score_21_22)

feat_21_22.to_csv(prep_dir + 'feat_h.csv', index=False)
print('Feature extraction completed. Files saved in '+ prep_dir)