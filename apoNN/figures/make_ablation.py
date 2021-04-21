import apoNN.src.data as apoData
import apoNN.src.utils as apoUtils
import apoNN.src.vectors as vectors
import apoNN.src.fitters as fitters
import apoNN.src.evaluators as evaluators
import apoNN.src.occam as occam_utils
import numpy as np
import random
import pathlib
import pickle
from ppca import PPCA
import apogee.tools.path as apogee_path
import matplotlib.pyplot as plt
apogee_path.change_dr(16)

###Setup

root_path = pathlib.Path(__file__).resolve().parents[2]/"outputs"/"data"

def standard_fitter(z,z_occam):
        """This fitter performs a change-of-basis to a more appropriate basis for scaling"""
        return fitters.StandardFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def simple_fitter(z,z_occam):
        """This is a simple fitter that just scales the dimensions of the inputed representation. Which is used as a baseline"""
        return fitters.SimpleFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 0.99*height,
                f"{height:.3f}",
                ha='center', va='bottom',fontsize=6)

def ablation_performance(Z,Z_occam):
    ev_standard = evaluators.StandardEvaluator(Z,Z_occam,leave_out=True,fitter_class=standard_fitter)
    ev_simple = evaluators.StandardEvaluator(Z,Z_occam,leave_out=True,fitter_class=simple_fitter)
    ev_empty = evaluators.StandardEvaluator(Z,Z_occam,leave_out=True,fitter_class=fitters.EmptyFitter)
    return ev_empty.weighted_average, ev_simple.weighted_average,ev_standard.weighted_average


###Hyperparameters

z_dim = 30 #PCA dimensionality

###
###

with open(root_path/"spectra"/"without_interstellar"/"cluster.p","rb") as f:
    Z_occam = pickle.load(f)    

with open(root_path/"spectra"/"without_interstellar"/"pop.p","rb") as f:
    Z = pickle.load(f)    

ablation_Z = ablation_performance(Z[:,:z_dim],Z_occam[:,:z_dim])

###
###

with open(root_path/"labels"/"full"/"cluster.p","rb") as f:
    Y_occam_full = pickle.load(f)    

with open(root_path/"labels"/"full"/"pop.p","rb") as f:
    Y_full = pickle.load(f)    

ablation_Y_full = ablation_performance(Y_full,Y_occam_full)

###
###

with open(root_path/"labels"/"core"/"cluster.p","rb") as f:
    Y_occam_core = pickle.load(f)    

with open(root_path/"labels"/"core"/"pop.p","rb") as f:
    Y_core = pickle.load(f)    

ablation_Y_core = ablation_performance(Y_core,Y_occam_core)

### Plotting######


plt.style.use('seaborn-colorblind')
plt.style.use('tex')


labels = ["Spectra","All abundances","Abundance subset"]
x = np.arange(len(labels))  # the label locations
width = 0.26  # the width of the bars

no_transformation = [ablation_Z[0],ablation_Y_full[0],ablation_Y_core[0]]
only_scaling = [ablation_Z[1],ablation_Y_full[1],ablation_Y_core[1]]
full_algorithm = [ablation_Z[2],ablation_Y_full[2],ablation_Y_core[2]]


save_path = root_path.parents[0]/"figures"/"ablation"
save_path.mkdir(parents=True, exist_ok=True)


fig, ax = plt.subplots(figsize =apoUtils.set_size(apoUtils.column_width))
rects1 = ax.bar(x-width, no_transformation, width, label='On raw')
rects2 = ax.bar(x, only_scaling, width, label='On scaled')
rects3 = ax.bar(x +width, full_algorithm, width, label='On transformed')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.title("Ablation study")

ax.set_ylabel('Doppelganger rate',fontsize=8)
ax.set_xticks(x)
ax.set_ylim([0,0.07])
ax.set_xticklabels(labels, fontsize=7)
ax.legend(frameon=True)

#plt.savefig(pathlib.Path(__file__).resolve().parents[2]/"")

plt.savefig(save_path/"ablation.pdf",format="pdf",bbox_inches='tight')
