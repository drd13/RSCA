import matplotlib.pyplot as plt
import numpy as np



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

