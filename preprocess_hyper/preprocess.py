import numpy as np
import pickle
from preprocess_hyper.prep_func import *

# Load the zipped images
zipped_h = pickle.load(open('data/ziped_h_21_22.pkl', 'rb'))

# Create a list of functions to apply to the images
functions = [(downsample_dataset, {'factor': 3}),
             (apply_savgol_filter, {'window_length': 5, 'polyorder': 3}),
             (slice_raster, {'start_band': 0, 'end_band': 80})]

# Apply the functions to the images
prep_h = apply_functions_to_images(zipped_h, functions)

# Save the processed images
pickle.dump(prep_h, open('data/preprocessed_data/prep_h.pkl', 'wb'))
print('zipping images completed!!! Files saved as data/preprocessed_data/prep_h.pkl')