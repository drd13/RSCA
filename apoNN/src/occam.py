from astropy.io import fits
import numpy as np


class Occam():
    def __init__(self):  
        self.clusters  = self.load_cluster()
        self.members = self.load_members()
        self.cluster_id = list(self.members[1].data.field("CLUSTER"))
        self.apogee_id = list(self.members[1].data.field("APOGEE_ID"))
        self.rv_prob = self.members[1].data.field("rv_prob")
        self.feh_prob = self.members[1].data.field("feh_prob")
        self.pm_prob = self.members[1].data.field("PM_PROB")
        self.cg_prob = self.members[1].data.field("CG_prob")


        
    def load_cluster(self):
        clusters_path = "/share/splinter/ddm/modules/turbospectrum/spectra/dr16/apogee/vac/apogee-occam/occam_cluster-DR16.fits"
        return fits.open(clusters_path)
    
    def load_members(self):
        members_path = "/share/splinter/ddm/modules/turbospectrum/spectra/dr16/apogee/vac/apogee-occam/occam_member-DR16.fits"
        return fits.open(members_path)
