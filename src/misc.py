''' Contains a number of miscellaneous functions
'''
import numpy as np
import pandas as pd

def mask_out(img, mask):
    '''multiplying each channel of the image with the mask'''
    return img * mask[..., np.newaxis]

def m_spec_with_zero(im):
    '''function to find mean including the zeros'''
    _,_,nb=im.shape
    return np.reshape(im,(-1,nb)).mean(0)

def m_spec(im):
    '''function to find mean ignoring the zeros'''
    _,_,nb=im.shape
    im_c = im.copy()
    im_c[im_c==0]=np.nan
    return np.nanmean(np.reshape(im_c,(-1,nb)),0)

def m_spec_with_mask(im,prep_im):
    '''This function calculates the mean across each channels for im ignoring the values with zero prep_im'''
    _,_,nb=im.shape
    im_c = im.copy()
    im_c[prep_im==0]=np.nan
    return np.nanmean(np.reshape(im_c,(-1,nb)),0)

def increment_column_suffix(df):
    """
    This function increments the numeric suffix in each column name of a DataFrame by 1.
    For example, if a column name is 'feature_1', it will be renamed to 'feature_2'.
    If a column name does not end with a number, it remains unchanged.

    Parameters:
    df (pandas.DataFrame): The DataFrame whose column names are to be updated.

    Returns:
    df (pandas.DataFrame): The DataFrame with updated column names.
    """
    new_columns = []
    for col in df.columns:
        parts = col.split('_')
        if parts[-1].isdigit():
            parts[-1] = str(int(parts[-1]) + 1)
            new_columns.append('_'.join(parts))
        else:
            new_columns.append(col)
    df.columns = new_columns
    return df

def remove_columns_ending_with_0(df):
    '''Removes all blue features from the dataframe'''
    cols_to_drop = [col for col in df.columns if col.endswith('_0')]
    df = df.drop(columns=cols_to_drop)
    return df