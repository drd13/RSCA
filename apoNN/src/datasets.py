import apogee.tools.read as apread
import matplotlib.pyplot as plt
import apogee.tools.path as apogee_path
from apogee.tools import bitmask

import random
import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class ApogeeDataset(Dataset):
    def __init__(self,allStar=None,filename=None,outputs = None):
        """
        allStar: 
            an allStar shape array containg those stars chosen for the dataset
        filename:
            string containing filename (not path!) of pickled dictionary. 
        outputs:
            a list specifying what items to return for dataset calls
        """
        self.filtered_bits =  self.set_filtered_bits()
        
        if outputs is None:
            self.outputs = ["aspcap","physical","idx"]
        else: 
            self.outputs = outputs
            
        if allStar:
            self.allStar = allStar
            self.pickled = False
        elif filename:
            self.dataset = self.load(filename)
            self.pickled = True
        else:
            raise Exception("need to specify one of allStar or filename")

        
            
    def to_dict(self):
        """generates a pickled version of the currently loaded dataset"""
        pickled_dataset = {}
        for item in self.outputs:
            data = []
            for idx in range(len(self)):
                output = self.get_requested_output(idx,item)
                data.append(output)
            pickled_dataset[item] = data
        return pickled_dataset
    
    def dump(self, savename):
        """dump a pickle equivalent to dataset. Useful for faster I/O"""
        dict_dataset = self.to_dict()
        with open(os.path.join("../../outputs/pickled_datasets",f"{savename}.p"), "wb") as f:
            pickle.dump(dict_dataset,f)
            
    def load(self,filename):
        """load a pickled version of dataset"""
        with open(os.path.join("../../outputs/pickled_datasets",f"{filename}.p"), "rb") as f:
            dict_dataset = pickle.load(f)
        
        return dict_dataset 
    
        
    def idx_to_prop(self,idx):
        return self.allStar[idx]["APOGEE_ID"],self.allStar[idx]["FIELD"], self.allStar[idx]["TELESCOPE"]
    
    def get_pseudonormalized(self,apogee_id,loc,telescope):
        return apread.aspcapStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=1)
    
    def get_apstar(self,apogee_id,loc,telescope):
        spec,_ = apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=1)
        if len(spec.shape)==2:
            spec = spec[0] #unclear whether I should be operating on 0
        return spec
    
    def get_mask(self,apogee_id,loc,telescope):
        return apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=3)[0]
    
    def idx_to_physical(self,idx):
        t_eff = self.allStar[idx]["Teff"]
        log_g = self.allStar[idx]["logg"]
        return torch.tensor([self.scale(t_eff,4000,5000),self.scale(log_g,1.5,3.)])
    
    def scale(self,t_eff,upper,lower):
        return (t_eff-lower)/(upper-lower)
    
    
    def set_filtered_bits(self,filtered_bits=None):
        if filtered_bits:
            return filtered_bits
        else:
            return [bitmask.apogee_pixmask_int('BADPIX'),
         bitmask.apogee_pixmask_int('CRPIX'),
         bitmask.apogee_pixmask_int('SATPIX'),
         bitmask.apogee_pixmask_int('UNFIXABLE'),
         bitmask.apogee_pixmask_int('PERSIST_HIGH'),
         bitmask.apogee_pixmask_int('NOSKY'),
         bitmask.apogee_pixmask_int('BADFLAT'),
         bitmask.apogee_pixmask_int('BADFLAT'),
         bitmask.apogee_pixmask_int('LITTROW_GHOST')]
    
    def filter_mask(self,mask,filtered_bits):
        """takes a bit mask and returns an array with those elements to be included and excluded from the representation."""
        mask_arrays = np.array([bitmask.bit_set(bit,mask).astype(bool) for bit in filtered_bits])
        filtered_mask = np.sum(mask_arrays,axis=0)==0
        return filtered_mask
    
    
    def __len__(self):
        return len(self.allStar)
    
    def get_requested_output(self,idx, item):
        if self.pickled:
            return self.dataset[item][idx]
        else:
            if item == "aspcap":
                apogee_id,loc,telescope = self.idx_to_prop(idx)
                spec,hed = self.get_pseudonormalized(apogee_id,loc,telescope)
                return torch.tensor(spec.astype(np.float32))
                
            elif item == "physical":
                physical_params = self.idx_to_physical(idx)
                return physical_params
                
            elif item == "idx":
                return idx
            
            elif item == "mask":
                apogee_id,loc,telescope = self.idx_to_prop(idx)
                masks = self.get_mask(apogee_id,loc,telescope)
                if len(masks.shape)==2:
                    mask = masks[0] #unclear whether I should be operating on 0
                else:
                    mask = masks
                bool_mask = self.filter_mask(mask,self.filtered_bits)
                return torch.tensor(bool_mask)
            
            elif item == "apstar":
                apogee_id,loc,telescope = self.idx_to_prop(idx)
                spec = self.get_apstar(apogee_id,loc,telescope)
                return torch.tensor(spec.astype(np.float32))/250
            
            else: 
                raise Exception(f"{item} is not a valid iterable")
                
        
    
    def __getitem__(self,idx):
        returned = []
        for item in self.outputs:
            returned.append(self.get_requested_output(idx,item))
        return tuple(returned)