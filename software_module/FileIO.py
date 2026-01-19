# ============= #
# Preliminaries # 
# ============= # 

import tomllib
import os
from os import path
import pickle

# Function to load analysis configurations
def loadConfig(config_file):
    with open(config_file, mode="rb") as fp:
        config = tomllib.load(fp)
    return config


# Function to create a directory if it does not exist
def makeDir(dirpath):
    if not path.exists(dirpath):
        os.makedirs(dirpath)


# Function to load networkx from pickle
def loadNetworkPKL(pkl_file):
    with open(pkl_file, mode="rb") as fp:
        networks = pickle.load(fp)
    return networks