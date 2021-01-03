"""Some random utils functions. Most are not actually useed in the paper"""

import torch
import torchvision
import pickle
import numpy as np
from pathlib import Path
import apogee.tools.path as apath
import apogee.spec.window as apwindow



def generate_loss_with_masking(loss):
    def loss_with_masking(x_pred,x_true,mask):
        return loss(x_pred[mask],x_true[mask]) #mask contains the inputs we want to keep.
    return loss_with_masking
    


def dump(item,filename):
    filepath = Path(__file__).parents[2].joinpath("outputs","pickled_misc",f"{filename}.p")
    with open(filepath,"wb") as f:
        pickle.dump(item,f)


def load(filename):
    filepath = Path(__file__).parents[2].joinpath("outputs","pickled_misc",f"{filename}.p")
    with open(filepath,"rb") as f:
        item = pickle.load(f)
    return item



def get_window(elem):
    current_apogee_redux = apath._APOGEE_REDUX
    apath.change_dr(12)
    start,end = apwindow.waveregions(elem,asIndex=True)
    apath._APOGEE_REDUX = current_apogee_redux
    return (start,end)

def get_lines(elem,asIndex=False):
    current_apogee_redux = apath._APOGEE_REDUX
    apath.change_dr(12)
    lines = apwindow.lines(elem,asIndex=asIndex)
    apath._APOGEE_REDUX = current_apogee_redux
    return lines

def get_mask_elem(elem,trimmed=0):
    """trimmed: number of indexes to trim off of each windows edges"""
    spec_mask =np.zeros(8575)
    start,end = get_window(elem)
    for i in range(len(start)):
        line_idx = np.arange(start[i],end[i])
        spec_mask[line_idx] = 1 
    return spec_mask    
