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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ApogeeDataset(Dataset):
    def __init__(self,allStar):
        """
        allStar: 
                an allStar shape array containg those stars chosen for the dataset
        """
        self.allStar = allStar
        self.return_mask = False
        self.filtered_bits =  [bitmask.apogee_pixmask_int('BADPIX'),
         bitmask.apogee_pixmask_int('CRPIX'),
         bitmask.apogee_pixmask_int('SATPIX'),
         bitmask.apogee_pixmask_int('UNFIXABLE'),
         bitmask.apogee_pixmask_int('PERSIST_HIGH'),
         bitmask.apogee_pixmask_int('NOSKY'),
         bitmask.apogee_pixmask_int('BADFLAT'),
         bitmask.apogee_pixmask_int('BADFLAT'),
         bitmask.apogee_pixmask_int('LITTROW_GHOST')]
        
    def idx_to_prop(self,idx):
        return self.allStar[idx]["APOGEE_ID"],self.allStar[idx]["FIELD"], self.allStar[idx]["TELESCOPE"]
    
    def get_pseudonormalized(self,apogee_id,loc,telescope):
        return apread.aspcapStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=1)
    
    def get_mask(self,apogee_id,loc,telescope):
        return apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=3)[0]
    
    def idx_to_physical(self,idx):
        t_eff = self.allStar[idx]["Teff"]
        log_g = self.allStar[idx]["logg"]
        return torch.tensor([self.scale(t_eff,4000,5000),self.scale(log_g,1.5,3.)])
    
    def scale(self,t_eff,upper,lower):
        return (t_eff-lower)/(upper-lower)
    
    def filter_mask(self,mask,filtered_bits):
        """takes a bit mask and returns an array with those elements to be included and excluded from the representation."""
        mask_arrays = np.array([bitmask.bit_set(bit,mask).astype(bool) for bit in filtered_bits])
        filtered_mask = np.sum(mask_arrays,axis=0)==0
        return filtered_mask
    
    
    def __len__(self):
        return len(self.allStar)
    
    def __getitem__(self,idx):
        apogee_id,loc,telescope = self.idx_to_prop(idx)
        spec,hed = self.get_pseudonormalized(apogee_id,loc,telescope)
        physical_params = self.idx_to_physical(idx)
        if self.return_mask:
            masks = self.get_mask(apogee_id,loc,telescope)
            if len(masks.shape)==2:
                mask = masks[0] #unclear whether I should be operating on 0
            else:
                mask = masks
            bool_mask = self.filter_mask(mask,self.filtered_bits)
            return torch.tensor(spec.astype(np.float32)),torch.tensor(bool_mask),physical_params,idx
        else:
            return torch.tensor(spec.astype(np.float32)),physical_params,idx
        
