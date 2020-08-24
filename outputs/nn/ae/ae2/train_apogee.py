import apogee.tools.read as apread
import matplotlib.pyplot as plt
import apogee.tools.path as apogee_path
from apogee.tools import bitmask

import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from apoNN.src.datasets import ApogeeDataset
from tagging.src.networks import ConditioningAutoencoder,Embedding_Decoder,Feedforward,ParallelDecoder,Autoencoder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
apogee_path.change_dr(16)



########################################
###### Dataset Loading #################
########################################


allStar= apread.allStar(rmcommissioning=True,main=False,ak=True, akvers='targ',adddist=False)
upper_temp_cut = allStar["Teff"]<5000
lower_temp_cut = allStar["Teff"]>4000
lower_g_cut = allStar["logg"]>1.5
upper_g_cut = allStar["logg"]<3
snr_cut = allStar["SNR"]>100
snr_highcut = allStar["SNR"]<500
feh_outliercut = allStar["Fe_H"]>-5

combined_cut = lower_g_cut & upper_g_cut & lower_temp_cut & upper_temp_cut & snr_cut & snr_highcut & feh_outliercut
cut_allStar = allStar[combined_cut]


#######################################
##### Hyperparameter Setup ############
#######################################

n_batch = 64
n_z = 40
n_bins = 8575
lr = 0.0001
n_datapoints = 10000


########################################
##### Initialization ###################
########################################


dataset = ApogeeDataset(cut_allStar[:n_datapoints])
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = n_batch,
                                     shuffle= True,
                                     drop_last=True)



encoder = Feedforward([n_bins,2048,512,128,n_z],activation=nn.SELU()).to(device)
decoder = Feedforward([n_z,512,2048,8192,n_bins],activation=nn.SELU()).to(device)

autoencoder = Autoencoder(encoder,decoder,n_bins=n_bins).to(device)
optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=lr)


loss = nn.L1Loss()

#######################################
####### Training Loop #################
#######################################


for i in range(20000):
    if i%100==0:
        torch.save(autoencoder,f"ae_{i}.p")
        
    for j,(x,u,idx) in enumerate(loader):

        optimizer_autoencoder.zero_grad()
        x_pred,z = autoencoder(x.to(device))

        err_pred = loss(x_pred,x.to(device))

        err_tot = err_pred
        err_tot.backward()
        optimizer_autoencoder.step()
        if j%100==0:
            print(f"err:{err_tot},err_pred:{err_pred}")





