import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tifffile import tifffile
from skimage import transform as transf
from scipy.interpolate import interp1d

#loading tiff files
def load_im(fname, lb, ub, int=False, flt=False, rot_angle=-4.1):
    '''loading tiff files, where lb represents the lower band limit and ub the upper'''
    '''rot_angle represents the rotation angles in degrees for the image. I used -4.1 for 2021 and 2022 Geneva beets data'''
    im = tifffile.imread(fname)[:,:,lb:ub]
    im = transf.rotate(im,angle=rot_angle,mode='edge') #https://numpy.org/doc/stable/reference/generated/numpy.pad.html#numpy.pad
    im = np.clip(im,a_min=0,a_max=None) #clipping all the negative anamolous values to zero
    if not int: im=im/100
    if flt: im = im.reshape(-1,ub-lb)
    return im

#visualisation function
def con_stretch(img):return (img-np.min(img))/(np.max(img)-np.min(img))
def plot_rgb(tfile,rb=108,gb=68,bb=32):
    '''rb, gb and bb are the band numbers corresponding to red, green and blue respectively'''
    plt.imshow(con_stretch(tfile[:,:,[rb,gb,bb]]))
    return

def interpolate_dataframe(df, new_columns, method='quadratic', fill_value='extrapolate'):
    """
    This function interpolates a DataFrame along its columns using a specified method.

    Parameters:
    df (pd.DataFrame): The input DataFrame to interpolate. The DataFrame's columns are assumed to be numeric.
    new_columns (array-like): The new column values for which to interpolate the DataFrame's values.
    method (str, optional): The interpolation method to use. Defaults to 'quadratic'.
    fill_value (str or float, optional): The value to use for extrapolation when the new column values are outside the range of the original column values. Defaults to 'extrapolate', which means to use the method to extrapolate.

    Returns:
    pd.DataFrame: A new DataFrame with the same index as the input DataFrame and the new column values. The values are interpolated from the input DataFrame using the specified method. Any interpolated values that are negative are replaced with zero. Any interpolated values that are greater than 100 are capped at the maximum value in the original data that is not greater than 100.
    """
    # Convert the columns to integer type
    df.columns = df.columns.astype(int)

    # Create an interpolation function for each row
    f = interp1d(df.columns, df.values, kind=method, fill_value=fill_value, bounds_error=False, axis=1)

    # Apply the interpolation function to the new columns
    interpolated_values = f(new_columns)
    
    # Replace any negative values with zero
    interpolated_values = np.maximum(interpolated_values, 0)
    
    # Find the maximum value in the original data that is not greater than 100
    max_value_not_above_100 = df.values[df.values <= 100].max()

    # Cap any values greater than 100 at max_value_not_above_100
    interpolated_values = np.where(interpolated_values > 100, max_value_not_above_100, interpolated_values)

    df_interpolated = pd.DataFrame(interpolated_values, index=df.index, columns=new_columns)

    return df_interpolated