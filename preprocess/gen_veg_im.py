'''
This module is used for extracting vegetation indices from  zipped multispectral images. 
'''
import numpy as np
import pandas as pd
from src.misc import *
from src.utils import *
from src.veg_indices import *
from skimage import morphology
from skimage.transform import resize
from tqdm import tqdm
import pickle

file_save_dir = 'data/preprocessed_data/'  # directory to save the vegetation masked images
zipped_im21_22 = pickle.load(open('data/ziped_21_22.pkl', 'rb'))
zipped_im23 = pickle.load(open('data/ziped_23.pkl', 'rb'))

# split based on keys
zipped_im21 = {k: v for k, v in zipped_im21_22.items() if k[:4] == '2021'}
zipped_im22 = {k: v for k, v in zipped_im21_22.items() if k[:4] == '2022'}

#For wavelength combinations
wave_comb21 = np.array([475,560,668,717,840])
wave_comb23 = np.array([560, 650, 730, 860])

lb21,ub21 =0,5
lb23,ub23 = 0,4

mask_flight21 = '20210715'
mask_flight22 = '20220726'
mask_flight23 = '20230821'

def veg_extract(zipped_im, mask_flt_date, wave_comb, disk_param, black_region_width, thresh=0.25):
    '''
    This function takes a dictionary of images, a flight date for the mask, a wavelength combination, a disk parameter, a black region width, and a threshold value.
    '''
    # MSAVI2 used for thresholding
    msa = VI(wave_comb,840,670)
    zipped_veg_im = {}

    pbar = tqdm(total=len(zipped_im), desc=f'Extracting {mask_flt_date[:4]} images')
    for date in zipped_im.keys():
        zipped_veg_im[date] = {}
        for im_num in zipped_im[date].keys():
            
            ########## Skeleton generation code ############
            tfile = zipped_im[mask_flt_date][im_num]
            mask = (msa.MSA(tfile.T).T>thresh) # Thresholding for skeleton
            mask[:, :black_region_width] = 0
            mask[:, -black_region_width:] = 0
            mask2= morphology.opening(mask,morphology.disk(disk_param)) ###hyperparameter
            ################################################

            image = zipped_im[date][im_num]
            v_mask = msa.MSA(image.T).T > thresh # Thresholding the image
            veg_mask = v_mask & resize(mask2,v_mask.shape,mode='edge',anti_aliasing=False)  # the vegetation skeleton does not match the  vegetation v_mask
            zipped_veg_im[date][im_num] = mask_out(image, veg_mask)

            #print(date, im_num)
            #plot_rgb(zipped_veg_im[date][im_num], rb=2,gb=1,bb=0)
        pbar.update(1)
    pbar.close()
    return zipped_veg_im

zipped_veg_im21 = veg_extract(zipped_im21, mask_flight21, wave_comb21, disk_param = 7, black_region_width =9)
zipped_veg_im22 = veg_extract(zipped_im22, mask_flight22, wave_comb21, disk_param = 7, black_region_width =9)
pickle.dump({**zipped_veg_im21, **zipped_veg_im22}, open(file_save_dir+'veg_im21_22.pkl', 'wb'))

zipped_veg_im23 = veg_extract(zipped_im23, mask_flight23, wave_comb23, disk_param = 7, black_region_width =5)
pickle.dump(zipped_veg_im23, open(file_save_dir+'veg_im23.pkl', 'wb'))

print('Vegetation extraction complete, files saved in ' + file_save_dir)