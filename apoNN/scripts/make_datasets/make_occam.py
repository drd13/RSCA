import numpy as np
import matplotlib.pyplot as plt

import apogee.tools.read as apread
import apogee.tools.path as apogee_path
from apogee.tools import bitmask
import collections


from apoNN.src.datasets import ApogeeDataset
from apoNN.src.utils import generate_loss_with_masking,dump
import apoNN.src.vectors as vector
import apoNN.src.occam as occam_utils


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
apogee_path.change_dr(16)


occam = occam_utils.Occam()
allStar= apread.allStar()


occam_kept = occam.cg_prob>0.8

upper_temp_cut = allStar["Teff"]<5000
lower_temp_cut = allStar["Teff"]>4000
lower_g_cut = allStar["logg"]>1.5
upper_g_cut = allStar["logg"]<3

combined_cut = lower_g_cut & upper_g_cut & lower_temp_cut & upper_temp_cut 

occam_allStar,occam_cluster_idxs = occam_utils.prepare_occam_allStar(occam_kept,allStar[combined_cut])


registry = vector.OccamLatentVector.make_registry(occam_cluster_idxs)


one_star_cluster_ids = []
for key in registry:
        if len(registry[key])==1:
                    one_star_cluster_ids.append(registry[key][0])



multiple_star_cluster_ids = np.delete(np.arange(len(occam_allStar)),one_star_cluster_ids)


filtered_occam_allStar=  occam_allStar[multiple_star_cluster_ids]
filtered_occam_cluster_idxs  = occam_cluster_idxs[multiple_star_cluster_ids]


occam = {"allStar":filtered_occam_allStar,
         "cluster_idxs":filtered_occam_cluster_idxs}


dump(occam,"occam")
print("finnished dumping dataset")


