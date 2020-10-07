import torch
import torchvision
import pickle
from pathlib import Path

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




