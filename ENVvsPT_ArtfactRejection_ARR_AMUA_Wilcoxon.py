# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:04:54 2024

@author: shiyi
"""
import sys
import numpy as np
import numpy.matlib
import os 
# sys.path.append('C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\pipiline2.0\\EphysAnalysis')
import EphysAnalysis as ea
#%%
sig_path = ea.ask_directory()
ori_names = ea.get_names(sig_path, "_OriginSigArray.npy")
#%%
for ori_name in ori_names:
    ori_sig = ea.load_original(ori_name, sig_path)
    artifact_rejection = ea.ArtifactRejection(ori_sig)
    artifact_rejection.clean_array_ready()
    artifact_rejection.ARR_array_ready()
    for cc in range(artifact_rejection.nChannel):
        for ff in range(artifact_rejection.nRate):
            for dd in range(artifact_rejection.nDur):
                for ii in range(artifact_rejection.nITD):
                    for jj in range(artifact_rejection.nITD):
                        clean_sig = artifact_rejection.reject_artifact(cc, ff, dd, ii, jj)
                        
                        
                        
                        
                        
    