import apogee.tools.read as apread
import matplotlib.pyplot as plt
import apogee.tools.path as apogee_path
from apogee.tools import bitmask

import random
import numpy as np
import os
from pathlib import Path
import pickle

import apoNN.src.vectors as vector


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
        self.serialized=False
        
        if outputs is None:
            self.outputs = ["aspcap","aspcap_err","idx"]
        else: 
            self.outputs = outputs
            
        if allStar is not None:
            self.allStar = allStar
            self.dataset = self.to_dict()
            self.serialized = True


        elif filename:
            self.dataset = self.load(filename)
            self.serialized = True
        else:
            raise Exception("need to specify one of allStar or filename")
        
    def allStar_trimming(self,troublesome_ids):
        ids = np.arange(len(self.allStar))
        kept_ids = np.delete(ids,troublesome_ids)
        self.allStar = self.allStar[kept_ids]
            
    def to_dict(self):
        """generates a pickled version of the currently loaded dataset"""
        dict_dataset = {}
        troublesome_ids = []
        for item in self.outputs:
            data = []
            for idx in range(len(self)):
                try:
                    output = self.serialize(idx,item)
                    data.append(output)
                except:
                    print(f"\n\n\nFAILING IDX:{IDX}\n\n\n")
                    troublesome_ids.append(idx)
            dict_dataset[item] = np.array(data)
        self.allStar_trimming(troublesome_ids)
         
        return dict_dataset
    
    def dump(self, filename):
        """dump a pickle equivalent to dataset. Useful for faster I/O"""
        filepath = Path(__file__).parents[2].joinpath("outputs","pickled_datasets",f"{filename}.p")
        with open(filepath, "wb") as f:
            pickle.dump(self.dataset,f)
            
    def load(self,filename):
        """load a pickled version of dataset"""
        filepath = Path(__file__).parents[2].joinpath("outputs","pickled_datasets",f"{filename}.p")

        with open(filepath, "rb") as f:
            dict_dataset = pickle.load(f)
        return dict_dataset
    
        
    def idx_to_prop(self,idx):
        return self.allStar[idx]["APOGEE_ID"],self.allStar[idx]["FIELD"], self.allStar[idx]["TELESCOPE"]
    
    def get_aspcap(self,apogee_id,loc,telescope,ext=1):
        """returns aspcap spectra"""
        return apread.aspcapStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=ext)
    
    def get_apstar(self,apogee_id,loc,telescope,channel= 0):
        """returns apread spectra"""
        spec,_ = apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=1)
        if len(spec.shape)==2:
            spec = spec[channel] #unclear whether I should be operating on 0
        spec = spec/np.mean(spec)-1 #quick normalization of spectra
        return spec
    
    def get_mask(self,apogee_id,loc,telescope):
        return apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=3)[0]
    
    def idx_to_physical(self,idx,scale_physical=True):
        t_eff = self.allStar[idx]["Teff"]
        log_g = self.allStar[idx]["logg"]
        if scale_physical is True:
            return [self.rescale(t_eff,4000,5000),self.rescale(log_g,1.5,3.)]
        else:
            return [t_eff,log_g]
    
    def idx_to_parameters(self,idx,parameters,rescale=False):
        param_vals= np.array([self.allStar[idx][param] for param in parameters])
        if rescale is True:
            #not yet implemented
            return self.rescale(param_vals)
        else:
            return param_vals
    
    
    
    def rescale(self,param,upper,lower):
        return (param-lower)/(upper-lower)
    
    
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
    
    
    def interpolator(self,dataset,idx, filling_dataset=None):
        if filling_dataset is None:
            filling_dataset=dataset
            
        missing_values = dataset[idx]==0
        interpolating_candidates = filling_dataset[:,missing_values==False]
        similarity = np.sum((interpolating_candidates - dataset[idx,missing_values==False])**2,axis=1)
        similarity_argsort = list(similarity.argsort()) #1 because 0 is the spectra itself

        print(f"most similar:{similarity_argsort[1]}")
        spectra = np.copy(dataset[idx])
        zeroes_exist=True
        while zeroes_exist:
            most_similar_idx = similarity_argsort.pop(0)
            spectra[missing_values] = filling_dataset[most_similar_idx][missing_values] #while loop makes replacing with flagged ok
            missing_values = spectra==0
            print(np.sum(missing_values))
            if (missing_values==False).all():
                zeroes_exist=False

        return spectra         
        
    def serialize(self,idx,item):
        if item == "aspcap":
            apogee_id,loc,telescope = self.idx_to_prop(idx)
            spec,hed = self.get_aspcap(apogee_id,loc,telescope,ext=1)
            return spec.astype(np.float32)
        
        elif item == "aspcap_err":
            apogee_id,loc,telescope = self.idx_to_prop(idx)
            spec_err,_ = self.get_aspcap(apogee_id,loc,telescope,ext=2) 
            return spec_err.astype(np.float32)

        elif isinstance(item, list):
            parameters = self.idx_to_parameters(idx, item)
            return parameters

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
            return bool_mask

        elif item == "mask2":
            apogee_id,loc,telescope = self.idx_to_prop(idx)
            masks = self.get_mask(apogee_id,loc,telescope)
            if len(masks.shape)==2:
                mask = masks[1] #unclear whether I should be operating on 0
            else:
                mask = masks
            bool_mask = self.filter_mask(mask,self.filtered_bits)
            return bool_mask

        elif item == "apstar":
            apogee_id,loc,telescope = self.idx_to_prop(idx)
            spec = self.get_apstar(apogee_id,loc,telescope)
            return spec.astype(np.float32)

        elif item == "apstar2":
            apogee_id,loc,telescope = self.idx_to_prop(idx)
            spec = self.get_apstar(apogee_id,loc,telescope,channel=1)
            return spec.astype(np.float32)


        else: 
            raise Exception(f"{item} is not a valid iterable")
        
        
        
    def get_requested_output(self,idx, item):
            return self.dataset[item][idx]
            
    def __len__(self):
        if self.serialized:
            return len(self.dataset[list(self.dataset.keys())[0]])
        else:
            return len(self.allStar)
    
    
    def __getitem__(self,idx):
        returned = []
        for item in self.outputs:
            returned.append(self.get_requested_output(idx,item))
        return tuple(returned)


class ApstarDataset(ApogeeDataset):
    def __init__(self,allStar=None,filename=None, filling_dataset = None,tensor_type=torch.FloatTensor,recenter=False,continuum_normalize=None):
        """
        allStar: 
            an allStar shape array containg those stars chosen for the dataset
        filename:
            string containing filename (not path!) of pickled dictionary. 
        outputs:
            a list specifying what items to return for dataset calls
        """
        self.tensor = tensor_type
        self.filtered_bits =  self.set_filtered_bits()
        self.filling_dataset = filling_dataset
        self.err_threshold = 0.05
        self.serialized=False
        self.outputs = ["apstar","mask","idx"]
        self.recenter=recenter
        self.continuum_normalize = continuum_normalize
        if allStar is not None:
            self.allStar = allStar
            self.dataset = self.to_dict()
            self.serialized = True

        elif filename:
            self.dataset = self.load(filename)
            self.serialized = True
        else:
            raise Exception("need to specify one of allStar or filename")
            
        if self.recenter is True:
            self.x = vector.Vector(self.dataset["apstar_interpolated"]).centered
        else:
            self.x = vector.Vector(self.dataset["apstar_interpolated"]).raw
    def to_dict(self):
        """generates a pickled version of the currently loaded dataset and creates an aspcap_interpolated object containing interpolated spectra"""
        dict_dataset = ApogeeDataset.to_dict(self)
        
        interpolated_apstar = self.serialize_interpolation(dict_dataset)
        dict_dataset["apstar_interpolated"] = np.array(interpolated_apstar)
        return dict_dataset
    



class AspcapDataset(ApogeeDataset):
    def __init__(self,allStar=None,filename=None, filling_dataset = None,tensor_type=torch.FloatTensor,recenter=False):
        """
        allStar: 
            an allStar shape array containg those stars chosen for the dataset
        filename:
            string containing filename (not path!) of pickled dictionary. 
        outputs:
            a list specifying what items to return for dataset calls
        """
        self.tensor = tensor_type
        self.filtered_bits =  self.set_filtered_bits()
        self.filling_dataset = filling_dataset
        self.err_threshold = 0.05
        self.serialized=False
        self.outputs = ["aspcap","aspcap_err","idx"]
        self.recenter=recenter
        if allStar is not None:
            self.allStar = allStar
            self.dataset = self.to_dict()
            self.serialized = True

        elif filename:
            self.dataset = self.load(filename)
            self.serialized = True
        else:
            raise Exception("need to specify one of allStar or filename")
            
        
        if self.recenter is True:
            self.x = vector.Vector(self.dataset["aspcap_interpolated"]).centered
        else:
            self.x = vector.Vector(self.dataset["aspcap_interpolated"]).raw
        
    def to_dict(self):
        """generates a pickled version of the currently loaded dataset and creates an aspcap_interpolated object containing interpolated spectra"""
        dict_dataset = ApogeeDataset.to_dict(self)
        
        interpolated_aspcap = self.serialize_interpolation(dict_dataset)
        dict_dataset["aspcap_interpolated"] = np.array(interpolated_aspcap)

        return dict_dataset
    
    def serialize_interpolation(self,dict_dataset):
        data = []
        specs = dict_dataset["aspcap"]
        specs_err = dict_dataset["aspcap_err"]
        specs[specs_err>self.err_threshold] =0
        for idx in range(len(self)):
            print(idx)
            output = self.interpolate(specs,idx,self.filling_dataset)
            data.append(output)
        return data
        
    
    def interpolate(self,dataset,idx, filling_dataset=None):
        if filling_dataset is None:
            filling_dataset=dataset
        well_behaved_bins = np.sum(dataset,axis=0)!=0 #we want to ignore those bins always at zero
        missing_values = dataset[idx]==0
        interpolating_candidates = filling_dataset[:,missing_values==False]
        similarity = np.sum((interpolating_candidates - dataset[idx,missing_values==False])**2,axis=1)
        similarity_argsort = list(similarity.argsort()) #1 because 0 is the spectra itself
        spectra = np.copy(dataset[idx])
        zeroes_exist=True
        while zeroes_exist:
            most_similar_idx = similarity_argsort.pop(0)
            spectra[missing_values] = filling_dataset[most_similar_idx][missing_values] #while loop makes replacing with flagged ok
            missing_values = spectra==0
            if (missing_values[well_behaved_bins]==False).all():
                zeroes_exist=False

        return spectra         
        
        
    def get_requested_output(self,idx, item):
            return self.dataset[item][idx]
            
        
   
    def __len__(self):
        if self.serialized:
            return len(self.dataset[list(self.dataset.keys())[0]])
        else:
            return len(self.allStar)
    
    
    def __getitem__(self,idx):
        spectra = self.x[idx]    
        #self.get_requested_output(idx,"aspcap_interpolated")
        spectra_raw = self.get_requested_output(idx,"aspcap")
        spectra_err = self.get_requested_output(idx,"aspcap_err")

        idx = self.get_requested_output(idx,"idx")

        returned = [self.tensor(spectra),self.tensor(spectra_raw),self.tensor(spectra_err),torch.tensor(idx)]
        return tuple(returned)


