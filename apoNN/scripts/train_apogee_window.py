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

from apoNN.src.datasets import ApogeeDataset,AspcapDataset
from apoNN.src.utils import generate_loss_with_masking,get_mask_elem

from tagging.src.networks import ConditioningAutoencoder,Embedding_Decoder,Feedforward,ParallelDecoder,Autoencoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
apogee_path.change_dr(16)



#########################
##### Hyperparameters####
#########################
# These are all the hyperparameters used by the script


elem = "Mg"
mask_elem  = get_mask_elem(elem)
n_batch = 64
n_z = 5
n_bins = 8575
lr = 0.00001
dataset_name = "aspcap_training_clean"
loss = "l2"
use_masked_loss = True
savename = "ae"
encoder_architecture = [n_bins,2048,1024,n_z]
decoder_architecture = [n_z,256,512,np.sum(mask_elem).astype(int)]
activation = nn.LeakyReLU()
recenter=True

########################
### Setup ##############
########################



dataset = AspcapDataset(filename=dataset_name,recenter=recenter)
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = n_batch,
                                     shuffle= False,
                                     drop_last=True)


encoder = Feedforward(encoder_architecture ,activation=activation).to(device)
decoder = Feedforward(decoder_architecture ,activation=activation).to(device)
autoencoder = Autoencoder(encoder,decoder,n_bins=n_bins,intermediate_activation=activation).to(device)
optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=lr)


if loss == "l1":
    loss = nn.L1Loss()
elif loss == "l2":
    loss = nn.MSELoss()

if use_masked_loss:
    masked_loss = generate_loss_with_masking(loss)


t_mask_elem = torch.tensor(mask_elem).repeat(n_batch,1).type(torch.bool) #tensor for filtering

##############################
##### Training Loop ##########
##############################



for i in range(30000):
    if i%250==0:
        torch.save(autoencoder,f"{savename}_{i}.p")

    for j,(x,x_raw,x_err,idx) in enumerate(loader):
        optimizer_autoencoder.zero_grad()
        x_pred,z = autoencoder(x.to(device))
        x_window = x[t_mask_elem].reshape(n_batch,np.sum(mask_elem).astype(int))

        if use_masked_loss:
            mask_spec = x_err<dataset.err_threshold
            mask_spec__window = mask_spec[t_mask_elem].reshape(n_batch,np.sum(mask_elem).astype(int)) #only keep window of selected elem
            err_pred = masked_loss(x_pred,x_window.to(device),mask_spec_window)
        else:
            err_pred = loss(x_pred,x_window.to(device))

        err_tot = err_pred
        err_tot.backward()
        optimizer_autoencoder.step()
        if j%100==0:
            print(f"err:{err_tot},err_pred:{err_pred}")
                 

