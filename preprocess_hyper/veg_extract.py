import numpy as np
import pickle
from pysptools.classification import SAM
from src.misc import *
from preprocess_hyper.prep_func import *
from preprocess_hyper.veg_extract_func import *
from src.veg_indices import VI
import xarray as xr
from skimage import morphology, filters # type: ignore
import copy

#Load preprocessed images
prep_h = pickle.load(open('data/preprocessed_data/prep_h.pkl', 'rb'))

#Load wavelength bands
lb, ub = 0, 80
hyper_wave = np.linspace(398.573,1001.81,272)#list(float(re.sub('[a-zA-Z]', '', element)) for element in panels_h_21['20210715']['NBL'].long_name)
hyper_wave_3 = dn_sample_sig(hyper_wave, 3)[lb:ub]
multi_wave = [475,560,668,717,840]

# Generate vegetation binary map using RDVI
bin_map_gen = BinaryMapGenerator(
    distance_function = VI(hyper_wave_3,840,668).RD,
    threshold_function=filters.threshold_otsu, 
    threshold_direction = 'greater', 
    black_region_width=3, 
    morph_func=morphology.opening,
    morph_func_params={'footprint': morphology.disk(2)}
)
functions = [(bin_map_gen._generate_mask, {})]
bin_masks_h = apply_functions_to_images(prep_h, functions)

# Code for restricting the binary masks to remove weeds
flights = ['20220810', '20220818']
plots = [11, 19, 31]
bin_masks_h = apply_and_on_flights(bin_masks_h, '20220726', flights, plots)

# Extract vegetation using binary masks
veg_h = veg_extract_using_masks(prep_h, bin_masks_h)
# Save the processed images
pickle.dump(veg_h, open('data/preprocessed_data/veg_h.pkl', 'wb'))