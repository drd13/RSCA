import apoNN.src.utils as apoUtils
import apoNN.src.data as apoData
import apoNN.src.vectors as vector
import apoNN.src.occam as occam_utils

import apogee.tools.path as apogee_path
import apogee.tools.read as apread

import numpy as np
apogee_path.change_dr(16)


###### Hyperparameters
n_stars = 25000

###### AllStar

#allStar= apread.allStar(rmcommissioning=True,main=False,ak=True, akvers='targ',adddist=False)
#allStar_occam = apoUtils.load("occam")["allStar"]
#cluster_idxs = apoUtils.load("occam")["cluster_idxs"]
allStar = apoUtils.load("shuffled_allStar")


upper_temp_cut = allStar["Teff"]<5000
lower_temp_cut = allStar["Teff"]>4000
lower_g_cut = allStar["logg"]>1.5
upper_g_cut = allStar["logg"]<3.
occamlike_cut = lower_g_cut & upper_g_cut & lower_temp_cut & upper_temp_cut
allStar_occamlike =  allStar[np.where(occamlike_cut)]


occam = occam_utils.Occam()
occam_kept = occam.cg_prob>0.8
allStar_occam,cluster_idxs = occam_utils.prepare_occam_allStar(occam_kept,allStar_occamlike)





###### Create X

data_occamlike = apoData.Dataset(allStar_occamlike[0:n_stars])
data_occam = apoData.Dataset(allStar_occam)

X = apoData.infill_masked_spectra(data_occamlike.masked_spectra,data_occamlike.masked_spectra)
X_occam = apoData.infill_masked_spectra(data_occam.masked_spectra,data_occamlike.masked_spectra)

X = vector.Vector(X)
X_occam = vector.OccamLatentVector(raw = X_occam, cluster_names=cluster_idxs)

apoUtils.dump(X,"X2")
apoUtils.dump(X_occam,"X2_occam")
apoUtils.dump(allStar_occamlike,"allStar2")
