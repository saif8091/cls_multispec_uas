''' This file contains the functions for feature extraction from preprocessed images'''
import numpy as np
from src.texture import *
from src.veg_indices import *
from src.utils import *

def conv2value(im,init=0,final=np.nan):
    '''This function converts all value from init to final'''
    ''' I'm utilising this function for VIs.
    Convert the preprocessed images to np.nan for VI calculation
    this enables for VI calculation without warning
    '''
    im_c = im.copy()
    im_c[im_c==init] = final
    return im_c

def mask4tex(tex_stat, prep_im):
    '''This function converts all the  places where prep_im=0 to  nan'''
    ''' 
    Note I'm doing this instead of directly starting out with
    nan preprocessed images as the tex function cannot operate
    nan values. As such I create a nan mask after generating 
    descriptive statistics mask
    '''
    t_c = tex_stat.copy()
    t_c[prep_im==0] = np.nan
    return t_c

def calculate_stats(image):
    '''This function calculates the statistics of an image (h x w x ch)'''

    # Calculate various statistics for each channel, ignoring nan values
    mean = np.nanmean(image, axis=(0, 1))
    std = np.nanstd(image, axis=(0, 1))
    q1 = np.nanpercentile(image, 25, axis=(0, 1))  # First quartile
    q3 = np.nanpercentile(image, 75, axis=(0, 1))  # Third quartile
    skewness = np.nanmean((image - mean)**3, axis=(0, 1)) / std**3
    kurtosis = np.nanmean((image - mean)**4, axis=(0, 1)) / std**4 - 3

    # Return the results in a dictionary
    return {'mean': mean, 'cv': std/mean, 'q1': q1, 'q3': q3, 'skewness': skewness, 'kurtosis': kurtosis}

def features_from_single_veg_image(wave_comb, veg_image):
    '''
    This function calculates the features from a single masked images (non vegetation pixels are zero)
    Here, wave_comb is the wavelength combination for the image and veg_image is the masked image
    '''

    ## GLCM matrix initializer
    tex = fastglcm_wrapper(veg_image,levels=8,kernel_size=5,distance_offset=5,angles=[0,45,90,135])
    ## descriptive statistics calculation
    tex_feat_cont = calculate_stats(mask4tex(tex.calculate_glcm_contrast(),veg_image))
    tex_feat_entropy = calculate_stats(mask4tex(tex.calculate_glcm_entropy(),veg_image))
    tex_feat_homogenity = calculate_stats(mask4tex(tex.calculate_glcm_homogenity(),veg_image))
    tex_feat_asm = calculate_stats(mask4tex(tex.calculate_glcm_asm(),veg_image))
    tex_feat_dissimilarity = calculate_stats(mask4tex(tex.calculate_glcm_dissimilarity(),veg_image))
    tex_feat_mean = calculate_stats(mask4tex(tex.calculate_glcm_mean(),veg_image))
    tex_feat_var = calculate_stats(mask4tex(tex.calculate_glcm_var(),veg_image))
    tex_feat_cor = calculate_stats(mask4tex(tex.calculate_glcm_correlation(),veg_image))
    
    ## reflectance calculation
    veg_im_nan = conv2value(veg_image)
    reflectance_stats = calculate_stats(veg_im_nan)

    ## VI calculation
    clsi = calculate_stats(VI(wave_comb,668,570,734).Mahlein3idx(veg_im_nan.T,const=1).T)  # CLSI index Mahlein et al. (2013)
    hi = calculate_stats(VI(wave_comb,534,668,704).Mahlein3idx(veg_im_nan.T,const=0.5).T) # healthy index Mahlein et al. (2013)
    rdvi = calculate_stats(VI(wave_comb,840,668).RD(veg_im_nan.T).T)
    mcari2 = calculate_stats(VI(wave_comb,840,668,550).MCARI2(veg_im_nan.T).T)
    ngrdvi = calculate_stats(VI(wave_comb,550,668).ND(veg_im_nan.T).T)
    gvi = calculate_stats(VI(wave_comb,550,660,717,840).GVI(veg_im_nan.T).T)
    msavi2 = calculate_stats(VI(wave_comb,840,670).MSA(veg_im_nan.T).T)
    mcariosavi = calculate_stats(VI(wave_comb,840,670,717).MCARIOSAVI(veg_im_nan.T).T)

    return  {
            **{f'clsi_{key}': feature for key, feature in clsi.items()},
            **{f'hi_{key}': feature for key, feature in hi.items()},
            **{f'rdvi_{key}': feature for key, feature in rdvi.items()},
            **{f'mcari2_{key}': feature for key, feature in mcari2.items()},
            **{f'ngrdvi_{key}': feature for key, feature in ngrdvi.items()},
            **{f'gvi_{key}': feature for key, feature in gvi.items()},
            **{f'msavi2_{key}': feature for key, feature in msavi2.items()},
            **{f'mcariosavi_{key}': feature for key, feature in mcariosavi.items()},
            **{f'ref_{key}_{i}': feature for key, features in reflectance_stats.items() for i, feature in enumerate(features)},
            **{f'tex_cont_{key}_{i}': feature for key, features in tex_feat_cont.items() for i, feature in enumerate(features)},
            **{f'tex_ent_{key}_{i}': feature for key, features in tex_feat_entropy.items() for i, feature in enumerate(features)},
            **{f'tex_homo_{key}_{i}': feature for key, features in tex_feat_homogenity.items() for i, feature in enumerate(features)},
            **{f'tex_asm_{key}_{i}': feature for key, features in tex_feat_asm.items() for i, feature in enumerate(features)},
            **{f'tex_dis_{key}_{i}': feature for key, features in tex_feat_dissimilarity.items() for i, feature in enumerate(features)},
            **{f'tex_mean_{key}_{i}': feature for key, features in tex_feat_mean.items() for i, feature in enumerate(features)},
            **{f'tex_var_{key}_{i}': feature for key, features in tex_feat_var.items() for i, feature in enumerate(features)},
            **{f'tex_cor_{key}_{i}': feature for key, features in tex_feat_cor.items() for i, feature in enumerate(features)},
        }

import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_features(X_train, X_val):
    """
    Normalize features using StandardScaler for both the training and validation sets.

    Parameters:
        X_train (DataFrame or array-like): Training set features.
        X_val (DataFrame or array-like): Validation set features.

    Returns:
        X_train_normalized (DataFrame): Normalized training set features.
        X_val_normalized (DataFrame): Normalized validation set features.
        norm_model (StandardScaler): StandardScaler model used to normalize the data.
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform on the training set, then transform the validation set
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    
    # normalisation model
    norm_model = scaler
    # Convert the normalized NumPy arrays to DataFrames
    X_train_normalized = pd.DataFrame(X_train_normalized,index=X_train.index, columns=X_train.columns)
    X_val_normalized = pd.DataFrame(X_val_normalized, index=X_val.index, columns=X_val.columns)

    return X_train_normalized, X_val_normalized, norm_model