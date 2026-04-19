import os
import glob
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def create_temporal_tensor(folder_path, file_pattern="*_matrix.npy"):
    """
    Finds and stacks 2D adjacency matrices into a single 3D temporal tensor
    """
    search_path = os.path.join(folder_path, file_pattern)
    matrix_files = sorted(glob.glob(search_path))
    
    matrix_list = [np.load(f) for f in matrix_files]
    temporal_tensor = np.stack(matrix_list)

    return temporal_tensor


#%% Heatmap visualisation

plt.style.use('seaborn-v0_8-muted') 
num_frames = temporal_tensor.shape[0]
flat_tensor = temporal_tensor.reshape(num_frames, -1)

dist_matrix = squareform(pdist(flat_tensor, metric='correlation'))
similarity_matrix = 1 - dist_matrix
plt.figure(figsize=(9, 7))
im = plt.imshow(similarity_matrix, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)

cbar = plt.colorbar(im)
cbar.set_label('Network Similarity (Correlation)', rotation=270, labelpad=15)

plt.title("Hyper-Brain State Stability Over Time", fontsize=16)
plt.xlabel("Frame (time point)", fontsize=12)
plt.ylabel("Frame (time point)", fontsize=12)

plt.tight_layout()
plt.show()