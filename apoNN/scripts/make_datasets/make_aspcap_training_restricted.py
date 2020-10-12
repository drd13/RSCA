import apogee.tools.read as apread
import apogee.tools.path as apogee_path
from apogee.tools import bitmask
import numpy as np
import matplotlib.pyplot as plt


from apoNN.src.datasets import ApogeeDataset,AspcapDataset
from apoNN.src.utils import dump as dump 
from apoNN.src.utils import load as load 

apogee_path.change_dr(16)

allStar= apread.allStar(rmcommissioning=True,main=False,ak=True, akvers='targ',adddist=False)

upper_temp_cut = allStar["Teff"]<7000
lower_temp_cut = allStar["Teff"]>3500
lower_g_cut = allStar["logg"]>1.
upper_g_cut = allStar["logg"]<3.5
snr_cut = allStar["SNR"]>100
snr_highcut = allStar["SNR"]<500


training_cut = lower_g_cut & upper_g_cut & lower_temp_cut & upper_temp_cut & snr_cut & snr_highcut
training_cut_allStar = allStar[training_cut][:10000]

dataset = AspcapDataset(training_cut_allStar[:10000])
dataset.dump("aspcap_training_restricted")
dump(training_cut_allStar,"allStar_training_restricted")
