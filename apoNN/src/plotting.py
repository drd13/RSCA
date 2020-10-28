import matplotlib.pyplot as plt
import apoNN.src.vectors as vector
import numpy as np
import random


def summarize_representation(z_all, z_occam, percentage_all=0.1, percentage_occam=0.1,plot_cut=True):
    """Function to plot the histogram summary of a method
    z_all: np.array
        representation of full dataset
    z_occam: np.array
        representation of chemical clusters
    percentage_all: float
        Fraction of outliers in full dataset to cut. Used for result to not depend on outliers.
    percentage_occam: float
            Fraction of outliers in occam dataset to cut. Used for result to not depend on outliers.
    plot_cut: boolean
        whether to plot the raw or cut distribution
    """
    n_all = round(len(z_all)*percentage_all)
    n_occam = round(len(z_occam)*percentage_occam)
    sorted_all = z_all[np.abs(z_all).argsort()]
    sorted_occam = z_occam[np.abs(z_occam).argsort()]
    if n_occam!=0:
        var_occam = np.var(sorted_occam[:-n_occam])
    else:
        var_occam = np.var(sorted_occam)
    if n_all!=0:
        var_raw = np.var(sorted_all[:-n_all])
    else:
        var_raw = np.var(sorted_all)
    print(f"occam: {var_occam}")
    print(f"raw: {var_raw}")
    print(f"ratio: {var_raw/var_occam}")
    if plot_cut is True and n_all!=0 and n_occam!=0: 
        plt.hist(sorted_occam[:-n_occam],alpha=0.2,label="diff",bins=20,density=True)
        plt.hist(sorted_all[:-n_all],alpha=0.2,label="full",bins=20, density=True)
    else:
        print("here")
        plt.hist(sorted_occam,alpha=0.2,label="diff",bins=20,density=True)
        plt.hist(sorted_all,alpha=0.2,label="full",bins=20, density=True)



def get_combinations(len_cluster):
    """
    Given a cluster size returns a list containing every pair of index (without repeat) 
    """
    combinations = []
    for idx1 in np.arange(len_cluster):
        for idx2 in np.delete(np.arange(len_cluster),idx1):
            if sorted([idx1,idx2]) not in combinations:
                combinations.append(sorted([idx1,idx2]))
    return combinations



def get_intracluster_distances(z,z_occam,leave_out=True):
    """Measures intracluster distances (between stars in the same cluster) after fitting and transforming.
    INPUTS
    ------
    z: apoNN.vector.Vector
        A vector containing the field star dataset
    z_occam: vector.OccamLatentVector
        A vector containing the occam cluster stars
    leave_out: Boolean
        True corresponds to excluding clusters being evaluated from training so as to avoid overfitting 
    """
    distances = []
    for cluster in list(z_occam.registry.keys()):
        if leave_out is True:
            fitter = vector.Fitter(z,z_occam.without(cluster))
        else:
            fitter = vector.Fitter(z,z_occam)
        v_centered_occam = fitter.transform(z_occam.centered.only(cluster))
        combinations = get_combinations(len(v_centered_occam))
        distances_cluster = []
        for combination in combinations:
            distances_cluster.append(np.linalg.norm(v_centered_occam[combination[0]]-v_centered_occam[combination[1]]))

        distances.append(distances_cluster)
    return distances



def get_intercluster_distances(z,z_occam,leave_out=True,n_random = 200):
    """Measures intercluster distances (between stars in a cluster and stars from the field) after fitting and transforming
    INPUTS
    ------
    z: apoNN.vector.Vector
        A vector containing the field star dataset
    z_occam: vector.OccamLatentVector
        A vector containing the occam cluster stars
    leave_out: Boolean
        True corresponds to excluding clusters being evaluated from training so as to avoid overfitting 
    n_random: Number of field stars each cluste star is compared too.
    """
    distances = []
    for cluster in list(z_occam.registry.keys()):
        if leave_out is True:
            fitter = vector.Fitter(z,z_occam.without(cluster))
        else:
            fitter = vector.Fitter(z,z_occam)
        v_centered_occam = fitter.transform(z_occam.centered.only(cluster))
        v = fitter.transform(fitter.z.centered)
        n_v = len(v)
        distances_cluster = []
        for idx in np.arange(len(v_centered_occam)):
            for _ in np.arange(n_random):
                random_idx = random.randint(0,n_v-1)
                distances_cluster.append(np.linalg.norm(v_centered_occam[idx]-v[random_idx]))

        distances.append(distances_cluster)
    return distances

