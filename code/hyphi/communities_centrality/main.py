# -*- coding: utf-8 -*-

print("im working")

import pickle

# Data science libraries
#import numpy as np
#import pandas as pd
import networkx as nx
#from symengine import sin  # TODO: add to depedencies
#from jitcdde import jitcdde, t, y  # TODO: add to depedencies

# Visualization
import matplotlib.pyplot as plt

from hyphi.io import load_connectivity_data
from hyphi.analyses import build_sliding_window_graphs

#%%

#import sys
#sys.path.append("code/hyphi/spectral_analysis")

with open("01_prebase_graph.pkl", "rb") as f:
    data = pickle.load(f)

nx.draw(data, with_labels=True)
plt.show()

W, tract, roi_names, centers_raw, hemis_raw, areas_raw = load_connectivity_data("code/hyphi/spectral_analysis/01_prebase_graph.pkl")
#graphs = build_sliding_window_graphs(data, window_size=..., overlap=...)