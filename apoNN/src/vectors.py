import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Vector():
    def __init__(self, raw):
        self._raw = raw
    
    @property
    def raw(self):
        return self._raw
    
    @property
    def centered(self):
        return self._raw -np.mean(self._raw,axis=0)
    
    def zero_direction(self):
        """returns a new vector in which a direction has been zeroed out"""
        pass
    



class LatentVector(Vector):
    def __init__(self,  dataset, autoencoder, n_data = 100):
        self.autoencoder = autoencoder
        self.dataset = dataset
        self._raw = np.array([self.get_z(idx,self.dataset,self.autoencoder) for idx in range(n_data)]).squeeze()

    def get_z(self,idx,dataset,autoencoder):
        _,z = autoencoder(dataset[idx][0].to(device).unsqueeze(0))
        return z.detach().cpu().numpy()
    
    
    
class OccamLatentVector(Vector):
    def __init__(self,  dataset, autoencoder, occam_cluster_idxs, n_data = 100):
        self.autoencoder = autoencoder
        self.dataset = dataset
        self.occam_cluster_idxs = occam_cluster_idxs
        self._raw = np.array([self.get_z(idx,self.dataset,self.autoencoder) for idx in range(n_data)]).squeeze()

    def get_z(self,idx,dataset,autoencoder):
        _,z = autoencoder(dataset[idx][0].to(device).unsqueeze(0))
        return z.detach().cpu().numpy()
    
    
    @property
    def cluster_centered(self):
        z = np.zeros(self.raw.shape)
        for cluster_name in set(self.occam_cluster_idxs):
            cluster_idxs = np.where(self.occam_cluster_idxs==cluster_name)
            z[cluster_idxs]=self.raw[cluster_idxs]-self.raw[cluster_idxs].mean(axis=0)
        return z

 






class LinearTransformation():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    @property
    def val(self):
        return np.dot(self.y.centered.T,np.linalg.pinv(self.x.centered.T))
    
    def predict(self,vector:Vector):
        #need to return a Vector. So ncenteredeed to make this take the correct shape
        uncentered = np.dot(self.val,vector.centered.T).T
        centered = uncentered+np.mean(self.y.raw,axis=0)
        return Vector(centered)                
        #return np.dot(self.val,vector.centered.T)


