# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:04:54 2024

@author: shiyi
"""
import sys
import numpy as np
import numpy.matlib
import os 
sys.path.append('C:\\Users\\shiyi\\Documents\\1_Project\\ENVvsPT\\pipiline2.0\\EphysAnalysis')
import EphysAnalysis as ea
#%%
from tkinter import Tk
from tkinter.filedialog import askdirectory
root = Tk()
sig_path = askdirectory(title='Select Folder')
root.withdraw()
results_path = sig_path
file_names = os.listdir(sig_path)
ori_names = [file_name for file_name in file_names if all([x in file_name for x in ["_OriginSigArray.npy"]])]
print(ori_names)
#%%
for ori_name in ori_names:
    file = ea.AnalysisEphys(ori_name, sig_path)
    ori_sig = file.load_original()
    sig = ea.ArtifactRejection(ori_sig)
    sig.creat_clean_array()
    for cc in range(sig.nChannel):
        for ff in range(sig.nRate):
            for dd in range(sig.nDur):
                for ii in range(sig.nITD):
                    for jj in range(sig.nITD):
                        clean_sig = sig.artifact_rejection(cc, ff, dd, ii, jj)
                        
                        
                        
    