import numpy as np
from apoNN.src.occam import Occam
import apogee.tools.read as apread
import apogee.tools.path as apogee_path
import collections

from apoNN.src.datasets import ApogeeDataset,AspcapDataset
from apoNN.src.utils import generate_loss_with_masking,get_mask_elem
from apoNN.src.utils import dump
from apoNN.src.utils import load
import apoNN.src.vectors as vector
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
apogee_path.change_dr(16)

occam = load("occam")
allStar_occam = occam["allStar"]

dataset=  AspcapDataset(filename="aspcap_training_clean",tensor_type=torch.FloatTensor,recenter=True)
dataset_occam = AspcapDataset(allStar_occam,recenter=True,tensor_type=torch.FloatTensor,filling_dataset=dataset.dataset["aspcap"])
dataset_occam.dump("aspcap_occam")

