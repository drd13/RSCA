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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
apogee_path.change_dr(16)

###Setup

root_path = pathlib.Path(__file__).resolve().parents[2]/"outputs"/"data"
#root_path = pathlib.Path("/share/splinter/ddm/taggingProject/tidyPCA/apoNN/scripts").parents[1]/"outputs"/"data"

save_path = root_path.parents[0]/"figures"/"local"
save_path.mkdir(parents=True, exist_ok=True)



def standard_fitter(z,z_occam):
        """This fitter performs a change-of-basis to a more appropriate basis for scaling"""
        return fitters.StandardFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def simple_fitter(z,z_occam):
        """This is a simple fitter that just scales the dimensions of the inputed representation. Which is used as a baseline"""
        return fitters.SimpleFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)


###Hyperparameters

z_dim = 30 #PCA dimensionality
cmap = matplotlib.cm.get_cmap('viridis')
color1 = cmap(0.15)
color2 = cmap(0.75)


###
with open(root_path/"spectra"/"without_interstellar"/"cluster.p","rb") as f:
    Z_occam = pickle.load(f)

with open(root_path/"spectra"/"without_interstellar"/"pop.p","rb") as f:
    Z = pickle.load(f)



with open(root_path/"spectra"/"with_interstellar"/"cluster.p","rb") as f:
    Z_occam_interstellar = pickle.load(f)

with open(root_path/"spectra"/"with_interstellar"/"pop.p","rb") as f:
    Z_interstellar = pickle.load(f)



with open(root_path/"labels"/"core"/"cluster.p","rb") as f:
    Y_occam = pickle.load(f)

with open(root_path/"labels"/"core"/"pop.p","rb") as f:
    Y = pickle.load(f)

###calculate representations being visualized

evaluator_X = evaluators.StandardEvaluator(Z[:,:z_dim],Z_occam[:,:z_dim],leave_out=True,fitter_class=standard_fitter)
evaluator_X.weighted_average

evaluator_Y = evaluators.StandardEvaluator(Y,Y_occam,leave_out=True,fitter_class=standard_fitter)
evaluator_Y.weighted_average

### Make plots


#plot1
n_cols = 2
n_rows=6
start_idx = 0
fig = plt.figure(constrained_layout=True,figsize=[4*n_cols,2.5*n_rows])
gspec = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)
#for i in range(len(sorted(spectra_evaluator.registry))):
for i in range(n_rows):
    spec_ax = fig.add_subplot(gspec[i,0])
    evaluator_X.plot_cluster(sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2)
    abund_ax = fig.add_subplot(gspec[i,1])
    #abund_ax.set_xlabel("distance",fontsize=20)
    evaluator_Y.plot_cluster(sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2)

plt.savefig(save_path/"loc1.pdf",format="pdf")

#plot2
n_cols = 2
n_rows=6
start_idx = 6
fig = plt.figure(constrained_layout=True,figsize=[4*n_cols,2.5*n_rows])
gspec = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)
#for i in range(len(sorted(spectra_evaluator.registry))):
for i in range(n_rows):
    spec_ax = fig.add_subplot(gspec[i,0])
    evaluator_X.plot_cluster(sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2)
    abund_ax = fig.add_subplot(gspec[i,1])
    #abund_ax.set_xlabel("distance",fontsize=20)
    evaluator_Y.plot_cluster(sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2)
plt.savefig(save_path/"loc2.pdf",format="pdf")

#plot3
n_cols = 2
n_rows=5
start_idx = 12
fig = plt.figure(constrained_layout=True,figsize=[4*n_cols,2.5*n_rows])
gspec = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)
#for i in range(len(sorted(spectra_evaluator.registry))):
for i in range(n_rows):
    spec_ax = fig.add_subplot(gspec[i,0])
    evaluator_X.plot_cluster(sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2)
    abund_ax = fig.add_subplot(gspec[i,1])
    #abund_ax.set_xlabel("distance",fontsize=20)
    evaluator_Y.plot_cluster(sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2)
plt.savefig(save_path/"loc3.pdf",format="pdf")


#plot4
n_cols = 2
n_rows=5
start_idx = 17
fig = plt.figure(constrained_layout=True,figsize=[4*n_cols,2.5*n_rows])
gspec = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)
#for i in range(len(sorted(spectra_evaluator.registry))):
for i in range(n_rows):
    spec_ax = fig.add_subplot(gspec[i,0])
    evaluator_X.plot_cluster(sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2)
    abund_ax = fig.add_subplot(gspec[i,1])
    #abund_ax.set_xlabel("distance",fontsize=20)
    evaluator_Y.plot_cluster(sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2)
plt.savefig(save_path/"loc4.pdf",format="pdf")
    
