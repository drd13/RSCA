import numpy as np
import matplotlib.pyplot as plt
import torch

from tagging.src.networks import *
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
    
    @property
    def normalized(self):
        return self.centered/np.max(self.centered,0)
       



class LatentVector(Vector):
    def __init__(self,  dataset, autoencoder, n_data = 100):
        self.autoencoder = autoencoder
        self.dataset = dataset
        self._raw = np.array([self.get_z(idx) for idx in range(n_data)]).squeeze()

    def get_z(self,idx):
        _,z = self.autoencoder(self.dataset[idx][0].to(device).unsqueeze(0))
        return z.detach().cpu().numpy()
    
    
    def get_x(self,idx):
        return self.dataset[idx][0]
    
    def get_x_pred(self,idx):
        x_pred,_ = self.autoencoder(self.dataset[idx][0].to(device).unsqueeze(0))
        return x_pred.squeeze()
    
    def plot_rec(self,idx):
        plt.plot(z_occam.get_x_pred(clust_idxs[0][idx]).detach().cpu().numpy())
        plt.plot(z_occam.get_x(clust_idxs[0][idx]).detach().cpu().numpy())
        plt.xlim(4000,4200)
        
    
    
class OccamLatentVector(LatentVector):
    def __init__(self,  dataset, autoencoder, occam_cluster_idxs, n_data = 100):
        self.autoencoder = autoencoder
        self.dataset = dataset
        self.occam_cluster_idxs = occam_cluster_idxs
        self._raw = np.array([self.get_z(idx) for idx in range(n_data)]).squeeze()
    
    
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
        
        
        
class NonLinearTransformation():
    """transformation going from latent to neural network parameters"""
    def __init__(self,x,y):
        self.x = x
        self.y = y
        structure = [x.centered.shape[1],256,256,y.centered.shape[1]]
        self.network  = Feedforward(structure,activation=nn.SELU()).to(device)
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=0.0001)
        self.idx_loader = torch.utils.data.DataLoader(torch.arange(y.centered.shape[0]),batch_size=100)

        
    def fit(self,n_epochs = 20):
        for epoch in range(n_epochs):
            for idx in self.idx_loader:
                self.optimizer.zero_grad()
                x = torch.tensor(self.x.centered[idx]).to(device)
                y = torch.tensor(self.y.normalized[idx]).to(device)
                y_pred = self.network(x)
                err = self.loss(y_pred,y)
                err.backward()
                self.optimizer.step()
                print(f"err:{err}")
        return
    
    def predict(self,vector:Vector):
        #need to return a Vector. So ncenteredeed to make this take the correct shape
        x = torch.tensor(vector.centered).to(device)
        y_unscaled_pred = self.network(x).detach().cpu().numpy()
        y_pred = y_unscaled_pred*np.max(self.y.centered,0)+np.mean(self.y.raw,axis=0)
        return Vector(y_pred)                
        #return np.dot(self.val,vector.centered.T)



