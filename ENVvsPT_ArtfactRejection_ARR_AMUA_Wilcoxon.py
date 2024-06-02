# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:04:54 2024

@author: shiyi
"""
from matplotlib import pyplot as plt
import sys
import numpy as np
import numpy.matlib
import os 
# sys.path.append('C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\pipiline2.0\\EphysAnalysis')
import EphysAnalysis as ea
#%%
sig_path = ea.ask_directory()
# sig_path = 'D:/0_Project/0_ENVvsPT_Ephys/Data/ND/2022_01_24'
ori_names = ea.get_names(sig_path, "_OriginSigArray.npy")
#%%
for ori_name in ori_names:
    ori_sig = ea.load_original(ori_name, sig_path)
    artifact_rejection = ea.ArtifactRejection(ori_sig)
    artifact_rejection.clean_array_ready()
    artifact_rejection.ARR_array_ready()
    AMUA_calculation = ea.AMUACalc()
    for cc in range(artifact_rejection.nChannel):    
        for ff in range(artifact_rejection.nRate):
            for dd in range(artifact_rejection.nDur):
                for ii in range(artifact_rejection.nITD):
                    for jj in range(artifact_rejection.nITD):
                        '''
                        artifact rejection
                        '''
                        clean_sig = artifact_rejection.reject_artifact(cc, ff, dd, ii, jj)
                        artifact_rejection.calc_ARR(cc, ff, dd, ii, jj)
                        artifact_rejection.calc_ARRmean(cc, ff, dd, ii, jj)
                        '''
                        AMUA
                        '''
                        AMUA_calculation.calc_AMUA(clean_sig, cc, ff, dd, ii, jj)
                        
                        
                        
                        
                        
                        
    