import numpy as np
import pandas as pd
import os
import pickle
from src.utils import *

def zip_images(r_dir, lb, ub, rot_angle):
    """
    This function takes a directory of images, a lower and upper(5 for 2021/2022 and 4 for 2023) band numbers, and a rotation angle (-4.1 for 2021/2022, 0 for 2023). 
    It loads each image, applies the bounds and rotation, and then stores the image in a dictionary.
    The dictionary is structured with flight dates as keys, each containing another dictionary with plot numbers as keys and the corresponding image as the value.

    Parameters:
    r_dir (str): The directory where the images are stored.
    lb (int): The lower bound to apply to the image.
    ub (int): The upper bound to apply to the image.
    rot_angle (int): The rotation angle to apply to the image.

    Returns:
    plot_im_all (dict): A dictionary containing the processed images, structured by flight date and plot number.
    Example usage: plot_im_all['20210707'][11] to access the image for flight date 20210707, plot 11.
    """
    plot_im_all = {}
    count = 0
    for rfile in os.listdir(r_dir):
        rpath = os.path.join(r_dir,rfile)
        im = load_im(rpath,lb=lb,ub=ub,int=True,rot_angle=rot_angle)
        plt_num, flt_date = rfile.split('_')
        flt_date = flt_date.removeprefix('r').removesuffix('.tif')
        plt_num = int(plt_num)
        if flt_date not in plot_im_all:
            plot_im_all[flt_date] = {}
        plot_im_all[flt_date][plt_num] = im
        count += 1
    print(f'zipped {count} images')
    return plot_im_all

## ziping 2021/2022 images
dict_21_22 = zip_images('data/multispec_2021_2022', lb=0, ub=5, rot_angle=-4.1)
pickle.dump(dict_21_22, open('data/ziped_21_22.pkl', 'wb'))

## ziping 2023 images
dict_23 = zip_images('data/multispec_2023', lb=0, ub=4, rot_angle=0)
pickle.dump(dict_23, open('data/ziped_23.pkl', 'wb'))

print('Image zipping complete! Files can be found in the data directory.')