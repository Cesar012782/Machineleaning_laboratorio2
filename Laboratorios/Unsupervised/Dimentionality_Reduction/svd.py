# *****************************
#     SVD Module
# *****************************
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from unsupervised.utils.data_generation import generate_sample_data

def apply_svd(data, output_folder, sv_num_list):
    # Perform SVD and visualize progressively reconstructed images
    for sv_num in sv_num_list:
        reconstructed_data = svd_reconstruction(data, sv_num)

        # Save the reconstructed data
        output_path = os.path.join(output_folder, f'reconstructed_svd_{sv_num}.jpg')
        cv2.imwrite(output_path, reconstructed_data)

def svd_reconstruction(data, sv_num):
    u, s, v = np.linalg.svd(data, full_matrices=False)
    s[sv_num:] = 0  # Keep only the first 'sv_num' singular values
    reconstructed_data = np.dot(u, np.dot(np.diag(s), v))
    return np.uint8(reconstructed_data)