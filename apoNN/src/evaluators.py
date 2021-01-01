import abc
import matplotlib.pyplot as plt
import apoNN.src.vectors as vectors
import apoNN.src.fitters as fitters
import numpy as np
import random
import sklearn


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
        whether to plot the val or cut distribution
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
        var_val = np.var(sorted_all[:-n_all])
    else:
        var_val = np.var(sorted_all)
    print(f"occam: {var_occam}")
    print(f"val: {var_val}")
    print(f"ratio: {var_val/var_occam}")
    if plot_cut is True and n_all!=0 and n_occam!=0: 
        plt.hist(sorted_occam[:-n_occam],alpha=0.2,label="diff",bins=20,density=True)
        plt.hist(sorted_all[:-n_all],alpha=0.2,label="full",bins=20, density=True)
    else:
        print("here")
        plt.hist(sorted_occam,alpha=0.2,label="diff",bins=20,density=True)
        plt.hist(sorted_all,alpha=0.2,label="full",bins=20, density=True)




class Evaluator(abc.ABC):
    """Base class from which all evaluators are derived. This is an abstract class so is never called directly."""
    def __init__(self,z,z_occam,leave_out=True):
        pass
    
    
    @abc.abstractmethod
    def get_distances(self):
        pass
    

    @staticmethod
    def get_combinations(len_cluster):
        """
        Given a cluster size returns a list containing every index combination forming valid pairs (without repeat pairs) 
        """
        combinations = []
        for idx1 in np.arange(len_cluster):
            for idx2 in np.delete(np.arange(len_cluster),idx1):
                if sorted([idx1,idx2]) not in combinations:
                    combinations.append(sorted([idx1,idx2]))
        return combinations
    
    
    @staticmethod
    def get_doppelganger_rate(distances,random_distances,registry):
        """
        INPUTS
        distances: nested list or array containing intracluster distances. len of upper list should match number of clusters
        random_distance: nested list or array containing distances being compared to. len of upper list should match number of clusters
        registry: per-star cluster_identification"""
        doppelganger_rates = []
        stars_per_cluster = []
        for i in range(len(distances)):
            doppelganger_rate = np.mean(random_distances[i]<np.median(distances[i]))
            cluster_name = sorted(registry)[i]
            num_stars = len(registry[cluster_name])
            doppelganger_rates.append(doppelganger_rate)
            stars_per_cluster.append(num_stars)

        #weighted_average = np.average(doppelganger_rates,weights=stars_per_cluster)
        #average = np.average(doppelganger_rates)
        return doppelganger_rates
    
    
    @property
    def stars_per_cluster(self):
        """list containing number of stars in each cluster (clusters ordered in the same way as doppelganger rate)"""
        return [len(self.registry[cluster]) for cluster in sorted(self.registry)]
    
    @property
    def weighted_average(self):
        """average doppelganger rate where clusters are weighed based on number of stars"""
        return np.average(self.doppelganger_rates, weights=self.stars_per_cluster)
    
    @property
    def average(self):
        """average doppelganger rate where each cluster is weighted equally regardless of size"""
        return np.average(self.doppelganger_rates)
       
    
    @staticmethod
    def get_intracluster_distances(z,z_occam,leave_out=True,fitter_class=fitters.Fitter):
        """Measures intracluster distances (between stars in the same cluster) after fitting and transforming.
        INPUTS
        ------
        z: apoNN.vector.Vector
            A vector containing the field star dataset
        z_occam: vector.OccamVector
            A vector containing the occam cluster stars
        leave_out: Boolean
            True corresponds to excluding clusters being evaluated from training so as to avoid overfitting
        fitter_class: fitters.Fitter
            Fitter class that is used for rescaling.
        """
        distances = []
        for cluster in sorted(z_occam.registry):
            if leave_out is True:
                fitter = fitter_class(z,z_occam.without(cluster))
            else:
                fitter = fitter_class(z,z_occam)
            v_centered_occam = fitter.transform(z_occam.centered().only(cluster)).val
            combinations = Evaluator.get_combinations(len(v_centered_occam))
            distances_cluster = []
            for combination in combinations:
                distances_cluster.append(np.linalg.norm(v_centered_occam[combination[0]]-v_centered_occam[combination[1]]))

            distances.append(distances_cluster)
        return distances

    
    @staticmethod
    def get_intercluster_distances(z,z_occam,leave_out=True,n_random = 200,fitter_class=fitters.Fitter):
        """Measures intercluster distances (between stars in a cluster and stars from the field) after fitting and transforming
        INPUTS
        ------
        z: apoNN.vector.Vector
            A vector containing the field star dataset
        z_occam: vector.OccamVector
            A vector containing the occam cluster stars
        leave_out: Boolean
            True corresponds to excluding clusters being evaluated from training so as to avoid overfitting 
        n_random: int
            Number of field stars each cluster star is compared too.
        fitter_class: Fitter
            Fitter class that is used for rescaling.
        """
        distances = []
        for cluster in sorted(z_occam.registry):
            if leave_out is True:
                fitter = fitter_class(z,z_occam.without(cluster))
            else:
                fitter = fitter_class(z,z_occam)
            v_centered_occam = fitter.transform(z_occam.centered().only(cluster)).val
            v = fitter.transform(fitter.z.centered(z_occam)).val
            n_v = len(v)
            distances_cluster = []
            for idx in np.arange(len(v_centered_occam)):
                for _ in np.arange(n_random):
                    random_idx = random.randint(0,n_v-1)
                    distances_cluster.append(np.linalg.norm(v_centered_occam[idx]-v[random_idx]))

            distances.append(distances_cluster)
        return distances


    @staticmethod
    def get_field_distances(z,z_occam,leave_out=True,n_random = 20000,fitter_class=fitters.Fitter):
        """Measures intercluster distances between random field stars. This is an alternative random_distance approach to that found in get_intercluster_distance method
        INPUTS
        ------
        z: apoNN.vectors.Vector
            A vector containing the field star dataset
        z_occam: vectors.OccamVector
            A vector containing the occam cluster stars
        leave_out: Boolean
            True corresponds to excluding clusters being evaluated from training so as to avoid overfitting 
        n_random: Number of field stars each cluste star is compared too.
        use_relative_scaling: Boolean
            Whether to use a relative scaling in the fitter object.
        fitter_class: Fitter
            Fitter class that is used for rescaling.
        """
        distances = []
        for cluster in sorted(z_occam.registry):
            if leave_out is True:
                fitter = fitter_class(z,z_occam.without(cluster))
            else:
                fitter = fitter_class(z,z_occam)
            v = fitter.transform(fitter.z.centered(z_occam)).val
            n_v = len(v)
            distances_cluster = []
            for _ in np.arange(n_random):
                random_idx = random.randint(0,n_v-1)
                random_idx2 = random.randint(0,n_v-1)

                distances_cluster.append(np.linalg.norm(v[random_idx]-v[random_idx2]))
            distances.append(distances_cluster)

        return distances
    
    
    def plot_cluster(self,cluster_name,ax1=None,x_max=30):
        """function for visualizing the doppelganger rate of a chosen cluster.
        INPUTS
        ------
        cluster_name: string
            the name of the cluster (as found in the registry) to be plotted.
        ax1: matplotlib axis
            an axis on which to plot the cluster. Useful when combining subplots of individual clusters into one large plot.
        x_max: The cut-off to use on the x-axis of distances"""
        if ax1 is None:
            ax1 = plt.gca()
        index_cluster = sorted(self.registry).index(cluster_name)
        ax1.set_title(f"{cluster_name} ({self.stars_per_cluster[index_cluster]} stars), rate: {self.doppelganger_rates[index_cluster]:5.4f}")

        color1 = 'tab:blue'
        ax1.set_xlabel('distance')
        ax1.set_ylabel('p', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xlim(0,x_max)
        ax1.hist(self.distances[index_cluster],alpha=0.5,bins=self.stars_per_cluster[index_cluster],density=True,label="intercluster",color=color1)

        color2 = 'tab:orange'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.hist(self.random_distances[index_cluster],alpha=0.5,bins=200,density=True,label="random",color=color2)
        ax2.set_ylabel('p', color=color2)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.axvline(x=np.median(self.distances[index_cluster]),c="red",linestyle  = "--",linewidth=2)






class PcaEvaluator(Evaluator):
    def __init__(self,X,X_occam,n_components, leave_out = True,fitter_class = fitters.Fitter):
        """Class used for evaluating the doppelganger rate of a representation and visualizing it's performance. Applies PCA to the data before applying rescaling.
        INPUTS
        ------
        X: apoNN.vectors.Vector
            A vector containing the field star dataset
        X_occam: vectors.OccamVector
            A vector containing the occam cluster stars
        n_components: int
            how many PCA components to keep in the dimensionality reduction 
        leave_out: Boolean
            True corresponds to excluding clusters being evaluated from training so as to avoid overfitting

        """
        self.X = X
        self.X_occam = X_occam
        self.registry = X_occam.registry
        self.n_components = n_components
        self.leave_out = leave_out
        self.fitter_class = fitter_class
        self.distances,self.random_distances = self.get_distances(self.X,self.X_occam,self.n_components,self.leave_out,self.fitter_class)
        self.doppelganger_rates = self.get_doppelganger_rate(self.distances,self.random_distances,self.X_occam.registry)
        
    def get_distances(self,X,X_occam,n_components,leave_out,fitter_class):
        compressor = sklearn.decomposition.PCA(n_components=n_components,whiten=False)#z.val.shape[1],whiten=True)
        compressor.fit(X())
        Z  = vectors.Vector(compressor.transform(X()))
        Z_occam = vectors.OccamVector(cluster_names=X_occam.cluster_names, val= compressor.transform(X_occam()))

        distances = Evaluator.get_intracluster_distances(Z,Z_occam,fitter_class=fitter_class,leave_out=leave_out)
        random_distances = Evaluator.get_intercluster_distances(Z,Z_occam,n_random=1000,fitter_class=fitter_class,leave_out=leave_out)
        return distances,random_distances
    
    
class AbundanceEvaluator(Evaluator):
    def __init__(self,Y,Y_occam,leave_out=True,fitter_class = fitters.Fitter):
        """Class used for evaluating the doppelganger rate of a representation and visualizing it's performance. Directly rescales the inputed representation so is not suitable for low-dimensional data.
        INPUTS
        ------
        X: apoNN.vectors.Vector
            A vector containing the field star dataset
        X_occam: vectors.OccamVector
            A vector containing the occam cluster stars
        leave_out: Boolean
            True corresponds to excluding clusters being evaluated from training so as to avoid overfitting
        n_components: int
            how many PCA components to keep in the dimensionality reduction
        """
        self.Y = Y
        self.Y_occam = Y_occam
        self.registry = self.Y_occam.registry
        self.leave_out = leave_out
        self.fitter_class = fitter_class
        self.distances,self.random_distances = self.get_distances(self.Y,self.Y_occam,self.leave_out,fitter_class=fitter_class)
        self.doppelganger_rates = self.get_doppelganger_rate(self.distances,self.random_distances,self.Y_occam.registry)

    def get_distances(self,Y,Y_occam,leave_out,fitter_class):
        distances = Evaluator.get_intracluster_distances(Y,Y_occam,fitter_class=fitter_class,leave_out = leave_out)
        random_distances = Evaluator.get_intercluster_distances(Y,Y_occam,fitter_class=fitter_class,n_random=1000,leave_out = leave_out)
        return distances,random_distances
                
        
    
    
class PcaFieldEvaluator(PcaEvaluator):
    """Same as PCAEvaluator but calculates random distances using field-field pairs rather then cluster-field pairs"""
    def get_distances(self,X,X_occam,n_components,leave_out,fitter_class):
        compressor = sklearn.decomposition.PCA(n_components=n_components,whiten=False)#z.val.shape[1],whiten=True)
        compressor.fit(X())
        Z  = vectors.Vector(compressor.transform(X()))
        Z_occam = vectors.Occamector(cluster_names=X_occam.cluster_names, val= compressor.transform(X_occam()))

        distances = Evaluator.get_intracluster_distances(Z,Z_occam,fitter_class=fitter_class,leave_out=leave_out)
        random_distances = Evaluator.get_field_distances(Z,Z_occam,n_random=1000,fitter_class=fitter_class,leave_out=leave_out)
        return distances,random_distances


        

    
    

class AbundanceFieldEvaluator(AbundanceEvaluator):
    """Same as AbundanceEvaluator but calculates random distances using field-field pairs rather then cluster-field pairs"""

    def get_distances(self,Y,Y_occam,leave_out,fitter_class):
        distances = Evaluator.get_intracluster_distances(Y,Y_occam,fitter_class=fitter_class,leave_out = leave_out)
        random_distances = Evaluator.get_field_distances(Y,Y_occam,fitter_class=fitter_class,n_random=1000,leave_out = leave_out)
        return distances,random_distances

    

    
    
    
    
    
    
    
    
    
    
    
    
######## Experimental approach beware ################################    
    
    
    
    
class PcaAutoScalingEvaluator(PcaEvaluator):
    def get_distances(self,X,X_occam,n_components,leave_out=True):
        compressor = sklearn.decomposition.PCA(n_components=n_components,whiten=False)#z.val.shape[1],whiten=True)
        compressor.fit(X())
        Z  = vectors.Vector(compressor.transform(X()))
        Z_occam = vectors.OccamVector(cluster_names=X_occam.cluster_names, val= compressor.transform(X_occam()))

        distances = self.get_intracluster_distances(Z,Z_occam,use_relative_scaling=True,leave_out = leave_out)
        random_distances = self.get_intercluster_distances(Z,Z_occam,n_random=1000,use_relative_scaling=True,leave_out = leave_out)
        return distances,random_distances

    
    @staticmethod
    def get_intercluster_distances(z,z_occam,leave_out=True,n_random = 200,use_relative_scaling=True):
        """Measures intercluster distances (between stars in a cluster and stars from the field) after fitting and transforming
        INPUTS
        ------
        z: apoNN.vector.Vector
            A vector containing the field star dataset
        z_occam: vector.OccamLatentVector
            A vector containing the occam cluster stars
        leave_out: Boolean
            True corresponds to excluding clusters being evaluated from training so as to avoid overfitting 
        n_random: Number of field stars each cluster star is compared too.
        """
        distances = []
        for cluster in sorted(z_occam.registry):
            if leave_out is True:
                fitter = fitters.Fitter(z_occam.without(cluster),z_occam.without(cluster),use_relative_scaling=use_relative_scaling)
            else:
                fitter = fitters.Fitter(z_occam,z_occam,use_relative_scaling=use_relative_scaling)
            v_centered_occam = fitter.transform(z_occam.centered().only(cluster)).val
            v = fitter.transform(fitter.z.centered(z_occam)).val
            n_v = len(v)
            distances_cluster = []
            for idx in np.arange(len(v_centered_occam)):
                for _ in np.arange(n_random):
                    random_idx = random.randint(0,n_v-1)
                    distances_cluster.append(np.linalg.norm(v_centered_occam[idx]-v[random_idx]))

            distances.append(distances_cluster)
        return distances
    
    @staticmethod
    def get_intracluster_distances(z,z_occam,leave_out=True,use_relative_scaling=True):
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
        for cluster in sorted(z_occam.registry):
            if leave_out is True:
                fitter = fitters.Fitter(z_occam.without(cluster),z_occam.without(cluster),use_relative_scaling)
            else:
                fitter = fitters.Fitter(z_occam,z_occam,use_relative_scaling)
            v_centered_occam = fitter.transform(z_occam.centered().only(cluster)).val
            combinations = Evaluator.get_combinations(len(v_centered_occam))
            distances_cluster = []
            for combination in combinations:
                distances_cluster.append(np.linalg.norm(v_centered_occam[combination[0]]-v_centered_occam[combination[1]]))

            distances.append(distances_cluster)
        return distances

    
    
class AbundanceAutoScalingEvaluator(AbundanceEvaluator):
    def __init__(self,Y,Y_occam,leave_out=True):
        """Method evaluating doppelganger rate
        INPUTS
        ------
        X: apoNN.vector.Vector
            A vector containing the field star dataset
        X_occam: vector.OccamLatentVector
            A vector containing the occam cluster stars
        leave_out: Boolean
            True corresponds to excluding clusters being evaluated from training so as to avoid overfitting
        n_components: int
            how many PCA components to keep in the dimensionality reduction
        """
        self.Y = Y
        self.Y_occam = Y_occam
        self.registry = self.Y_occam.registry
        self.leave_out = leave_out
        self.distances,self.random_distances = self.get_distances(self.Y,self.Y_occam,self.leave_out)
        self.doppelganger_rates = self.get_doppelganger_rate(self.distances,self.random_distances,self.Y_occam.registry)

    def get_distances(self,Y,Y_occam,leave_out=True):
        distances = self.get_intracluster_distances(Y,Y_occam,use_relative_scaling=True,leave_out = leave_out)
        random_distances = self.get_intercluster_distances(Y,Y_occam,n_random=1000,use_relative_scaling=True,leave_out = leave_out)
        return distances,random_distances  
    
    
    @staticmethod
    def get_intercluster_distances(z,z_occam,leave_out=True,n_random = 200,use_relative_scaling=True):
        """Measures intercluster distances (between stars in a cluster and stars from the field) after fitting and transforming
        INPUTS
        ------
        z: apoNN.vector.Vector
            A vector containing the field star dataset
        z_occam: vector.OccamLatentVector
            A vector containing the occam cluster stars
        leave_out: Boolean
            True corresponds to excluding clusters being evaluated from training so as to avoid overfitting 
        n_random: Number of field stars each cluster star is compared too.
        """
        distances = []
        for cluster in sorted(z_occam.registry):
            if leave_out is True:
                fitter = fitters.Fitter(z_occam.without(cluster),z_occam.without(cluster),use_relative_scaling=use_relative_scaling)
            else:
                fitter = fitters.Fitter(z_occam,z_occam,use_relative_scaling=use_relative_scaling)
            v_centered_occam = fitter.transform(z_occam.centered().only(cluster)).val
            v = fitter.transform(fitter.z.centered(z_occam)).val
            n_v = len(v)
            distances_cluster = []
            for idx in np.arange(len(v_centered_occam)):
                for _ in np.arange(n_random):
                    random_idx = random.randint(0,n_v-1)
                    distances_cluster.append(np.linalg.norm(v_centered_occam[idx]-v[random_idx]))

            distances.append(distances_cluster)
        return distances
    
    @staticmethod
    def get_intracluster_distances(z,z_occam,leave_out=True,use_relative_scaling=True):
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
        for cluster in sorted(z_occam.registry):
            if leave_out is True:
                fitter = fitters.Fitter(z_occam.without(cluster),z_occam.without(cluster),use_relative_scaling)
            else:
                fitter = fitters.Fitter(z_occam,z_occam,use_relative_scaling)
            v_centered_occam = fitter.transform(z_occam.centered().only(cluster)).val
            combinations = Evaluator.get_combinations(len(v_centered_occam))
            distances_cluster = []
            for combination in combinations:
                distances_cluster.append(np.linalg.norm(v_centered_occam[combination[0]]-v_centered_occam[combination[1]]))

            distances.append(distances_cluster)
        return distances

