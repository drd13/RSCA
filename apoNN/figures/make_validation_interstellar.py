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
#root_path = pathlib.Path("/share/splinter/ddm/taggingProject/tidyPCA/apoNN/scripts").parents[1]/"outputs"/"data"


def standard_fitter(z,z_occam):
        """This fitter performs a change-of-basis to a more appropriate basis for scaling"""
        return fitters.StandardFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def simple_fitter(z,z_occam):
        """This is a simple fitter that just scales the dimensions of the inputed representation. Which is used as a baseline"""
        return fitters.SimpleFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def get_similarity(X,Y,n_repeats=50000,n_max=10000,use_delta=True):
    """
    OUTPUTS
    -------
    similarity_list: 
        contains the chemical similarity for random pairs of stars
    delta_list:
        contains the difference in variable of interest for these same stars
    use_delta: boolean
        if true give the difference between two varialbles. If false give the average.
    """
    assert len(X)==len(Y)
    similarity_list = []
    delta_list = []
    for _ in range(n_repeats):
        i,j = np.random.choice(n_max,2)
        if  (Y[i]>-999) and (Y[j]>-999): #hack to prevent pairs containing -9999 from showing up
            similarity_list.append(similarity_ij(i,j,X))
            if use_delta is True:
                delta_list.append(np.abs(Y[i]-Y[j]))
            else:
                delta_list.append(np.mean([Y[i],Y[j]]))
    return np.array(similarity_list),delta_list

def similarity_ij(i,j,v):
    return np.linalg.norm(v[i]-v[j])



###Hyperparameters

z_dim = 30 #PCA dimensionality


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

with open(root_path/"allStar.p","rb") as f:
    allStar = pickle.load(f)    


#### Apply metric learing #####


Z_fitter = standard_fitter(Z[:,:z_dim],Z_occam[:,:z_dim])
V = Z_fitter.transform(Z_fitter.z.centered(Z_occam[:,:z_dim]))



Z_fitter_interstellar = standard_fitter(Z_interstellar[:,:z_dim],Z_occam_interstellar[:,:z_dim])
V_interstellar = Z_fitter_interstellar.transform(Z_fitter.z.centered(Z_occam_interstellar[:,:z_dim]))



Y_fitter = simple_fitter(Y,Y_occam)
V_Y = Y_fitter.transform(Y.centered(Y_occam))


y_interest = allStar["AK_TARG"]

similarities,extinction_diffs = get_similarity(V.val,y_interest)
similarities_interstellar,extinction_diffs_interstellar = get_similarity(V_interstellar.val,y_interest)
similarities_y,extinction_diffs_y = get_similarity(V_Y.val,y_interest)

### Normalize similarities for sake of comparison
similarities = similarities/np.mean(similarities)
similarities_interstellar = similarities_interstellar/np.mean(similarities_interstellar)
similarities_y = similarities_y/np.mean(similarities_y)



try: 
    plt.style.use("tex")
except:
    print("tex style not implemented (https://jwalton.info/Embed-Publication-Matplotlib-Latex/)")
            
plt.style.use('seaborn')


fig, ax = plt.subplots(1, 3, figsize=apoUtils.set_size(apoUtils.text_width, fraction=2.5,subplots=(1, 3)))


ax[0].hist(similarities[extinction_diffs<np.median(extinction_diffs)],bins = 40,alpha=0.5,density=True,label="similar extinction pairs")
ax[0].hist(similarities[extinction_diffs>np.median(extinction_diffs)],bins = 40,alpha=0.5,density=True,label="dissimilar extinction pairs")
ax[0].axvline(x=np.mean(similarities[extinction_diffs<np.median(extinction_diffs)]))
ax[0].axvline(x=np.mean(similarities[extinction_diffs>np.median(extinction_diffs)]),color="green")

ax[0].set_xlabel("similarity")
ax[0].set_ylabel("$p$")
ax[0].set_title("from masked spectra")
ax[0].set_xlim([0,3])
ax[0].legend()

ax[1].hist(similarities_interstellar[extinction_diffs_interstellar<np.median(extinction_diffs_interstellar)],bins = 40,alpha=0.5,density=True,label="similar extinction pairs")
ax[1].hist(similarities_interstellar[extinction_diffs_interstellar>np.median(extinction_diffs_interstellar)],bins = 40,alpha=0.5,density=True,label="dissimilar extinction pairs")
ax[1].axvline(x=np.mean(similarities_interstellar[extinction_diffs_interstellar<np.median(extinction_diffs_interstellar)]))
ax[1].axvline(x=np.mean(similarities_interstellar[extinction_diffs_interstellar>np.median(extinction_diffs_interstellar)]),color="green")

ax[1].set_xlabel("similarity")
ax[1].set_ylabel("$p$")
ax[1].set_title("from full spectra")
ax[1].set_xlim([0,3])
ax[1].legend()

ax[2].hist(similarities_y[extinction_diffs_y<np.median(extinction_diffs_y)],bins = 20,alpha=0.5,density=True,label="similar extinction pairs")
ax[2].hist(similarities_y[extinction_diffs_y>np.median(extinction_diffs_y)],bins = 20,alpha=0.5,density=True,label="dissimilar extinction pairs")
ax[2].axvline(x=np.mean(similarities_y[extinction_diffs_y<np.median(extinction_diffs_y)]))
ax[2].axvline(x=np.mean(similarities_y[extinction_diffs_y>np.median(extinction_diffs_y)]),color="green")

ax[2].set_xlabel("similarity")
ax[2].set_ylabel("$p$")
ax[2].set_title("from stellar abundances")
ax[2].set_xlim([0,3])
ax[2].legend()

plt.savefig("validation_interstellar.pdf",format="pdf")
