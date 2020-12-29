import apogee.tools.read as apread
import matplotlib.pyplot as plt
import apogee.tools.path as apogee_path
from apogee.tools import bitmask

import random
import numpy as np

import apoNN.src.utils as apoUtils
from apoNN.src.data import Dataset

from tagging.src.networks import ConditioningAutoencoder,Embedding_Decoder,Feedforward,ParallelDecoder,Autoencoder

apogee_path.change_dr(16)


allStar= apoUtils.load("shuffled_allStar")





n_datapoints = 5000
n_start =25000
#dataset = ApogeeDataset(cut_allStar[:n_datapoints],outputs = ["aspcap","physical","idx"])
dataset = Dataset(allStar[n_start:n_datapoints+n_start])


for i in range(n_datapoints):
    print(i)
    try:
        dataset[i][1]
    except:
        pass





