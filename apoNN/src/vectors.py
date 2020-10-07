import numpy as np
import matplotlib.pyplot as plt
import torch

from tagging.src.networks import *
from astropy.io import fits

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def project(data,direction):
    """obtain linear projection of data along direction"""
    return np.dot(direction,data.T).squeeze()

def get_vars(data,directions):
    return np.var(project(data,directions),axis=1)

from sklearn.preprocessing import PolynomialFeatures

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
        return self.centered/np.max(np.abs(self.centered),0)
    



class LatentVector(Vector):
    def __init__(self,  dataset, autoencoder, n_data = 100):
        self.autoencoder = autoencoder
        self.dataset = dataset
        self._raw = np.array([self.get_z(idx) for idx in range(n_data)]).squeeze()

    def get_z(self,idx):
        _,z = self.autoencoder(torch.tensor(self.dataset[idx][0]).to(device).unsqueeze(0))
        return z.detach().cpu().numpy()
    
    
    def get_x(self,idx):
        return self.dataset[idx][0]
   
    def get_mask(self,idx):
        return self.dataset[idx][1]
    
    def get_x_pred(self,idx):
        x_pred,_ = self.autoencoder(torch.tensor(self.dataset[idx][0]).to(device).unsqueeze(0))
        return x_pred.squeeze().detach().cpu().numpy()
    
    def plot(self,idx,limits=[4000,4200]):
        plt.plot(self.get_x_pred(idx),label="pred")
        plt.plot(self.get_x(idx),label="real")
        plt.legend()
        plt.xlim(limits)
        
      
        
    
    
class OccamLatentVector(LatentVector):
    def __init__(self,  dataset, autoencoder, cluster_names, n_data = 100):
        LatentVector.__init__(self,dataset,autoencoder,n_data)
        self.cluster_names = cluster_names
        self.registry = self.make_registry(self.cluster_names)
    
    
    def make_registry(self,cluster_names):
        clusters = list(set(cluster_names))
        cluster_registry = {}
        for cluster in clusters:
            cluster_idxs = np.where(cluster_names==cluster)
            cluster_registry[cluster] = cluster_idxs[0]
        return cluster_registry
    
    
    @property
    def cluster_centered(self):
        z = np.zeros(self.raw.shape)
        for cluster in self.registry:
            cluster_idxs = self.registry[cluster]
            z[cluster_idxs]=self.raw[cluster_idxs]-self.raw[cluster_idxs].mean(axis=0)
        return z
    
    
    

class AstroNNVector(Vector):
    def __init__(self,allStar,params):
        self.astroNN_hdul = fits.open("/share/splinter/ddm/modules/turbospectrum/spectra/dr16/apogee/vac/apogee-astronn/apogee_astroNN-DR16-v0.fits")
        self.allStar = allStar
        self.params = params
        ids = self.get_astroNN_ids(self.allStar)
        cut_astroNN = self.astroNN_hdul[1].data[ids]
        self._raw = self.generate_abundances(cut_astroNN,self.params)
        
        
    def get_astroNN_ids(self,allStar):
        desired_ids= []
        astroNN_ids = list(self.astroNN_hdul[1].data["Apogee_id"])
        for apogee_id in allStar["APOGEE_ID"]:
            desired_ids.append(astroNN_ids.index(apogee_id))
        return desired_ids
    
    
    def generate_abundances(self,astroNN,params):
        values = []
        for i,p in enumerate(params):
            fe_h = astroNN["FE_H"]
            if p in ["Teff","logg","Fe_H"]:
                values.append(astroNN[p])
            else:
                p_h = astroNN[params[i].split("_")[0]+"_H"]
                values.append(p_h-fe_h)

        return np.array(values).T
 
    
    
  
    
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
        y_pred = y_unscaled_pred*np.max(np.abs(self.y.centered),0)+np.mean(self.y.raw,axis=0)
        return Vector(y_pred)                
        #return np.dot(self.val,vector.centered.T)


        
