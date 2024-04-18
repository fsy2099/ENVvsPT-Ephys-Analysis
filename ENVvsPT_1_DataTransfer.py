#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:40:41 2022

read data from RawDataPath
transfer the data from .ephys to .npy
save to results_path

@author: shiyi
"""
from sys import platform
import RZ2ephys as ep
import os
import numpy as np
from matplotlib import pyplot as plt
# give recording data path and saving path
if platform == "darwin": # if windows system  
    RawDataPath = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/' 
    results_path = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/' 
elif platform == "linux": # if linux system
    RawDataPath = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/'
    results_path = '/disks/CIdata/1_ENVvsPT/2_ND/2023_03_03/Results_Data/'
#%%
#position_num = 'P04'
file_names = os.listdir(RawDataPath)
sig_names = [file_name for file_name in file_names if all([x in file_name for x in ['_ENV_vs_PT', ".csv"]])]
for pp in range(len(sig_names)):

    fname = sig_names[pp][:-4]  
    swps, stm = ep.readEphysFile(RawDataPath+fname)
    if len(swps)-1 == len(stm):
        swps_new=np.delete(swps,0)
    else:
        continue        
    stiDur = np.sort(stm['duration (s)'].unique())
    stiRate = np.sort(stm['clickRate (Hz)'].unique())   
    stiITD = np.sort(stm['ITD (ms)'].unique())
    stienvITD = np.sort(stm['env ITD (ms)'].unique())
    
    Fs = swps[0].sampleRate
    sigSamples = []
    for ii in range(swps_new.shape[0]):
        tempSamples = swps_new[ii].signal
        sigSamples.append(tempSamples.shape[0])    
    nsamples = min(sigSamples)-10
    print(min(sigSamples)/Fs)
    if min(sigSamples)/Fs < 0.3:
        continue
    nchans = swps_new[0].signal.shape[1]
#    nsamples = int(Fs*0.4)
    ntrials = np.shape(np.array(stm[(stm['clickRate (Hz)'] == stiRate[0]) & (stm['duration (s)'] == stiDur[0]) & (stm['ITD (ms)'] == stiITD[0]) & (stm['env ITD (ms)'] == stienvITD[0])].index))[0] # count trial number
    signal_arrays = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD),nsamples,ntrials),dtype = 'float32')
    ErrorMark_array = np.zeros((nchans,len(stiRate),len(stiDur),len(stiITD),len(stienvITD)), dtype = 'float32')
    for cc in range(nchans):
        print('Chan'+str(cc+1))
        for ff in range(len(stiRate)):        
            for dd in range(len(stiDur)):
                for ii in range(len(stiITD)):
                    for jj in range(len(stienvITD)):
                        stimParam = [stiRate[ff],stiDur[dd],stiITD[ii],stienvITD[jj]]
                        print(stimParam)
                        stm_select = stm[(stm['clickRate (Hz)'] == stimParam[0]) & (stm['duration (s)'] == stimParam[1]) & (stm['ITD (ms)'] == stimParam[2]) & (stm['env ITD (ms)'] == stimParam[3])]
                        stmIdx = np.array(stm_select.index)
                        # check trial number, if number of trials is shoter than defalt (15 or 30), set error mark as 1
                        if len(stmIdx) < ntrials:
                            ErrorMark_array[cc, ff, dd, ii, jj] = 1
                            print(str(len(stmIdx))+' wrong! uneuqal trial length')                            
                            continue
                        elif len(stmIdx) > ntrials:
                            stmIdx = stmIdx[:ntrials]
                        # aline up each trial, use the first trial as reference
                        sampleIdx = int(Fs*stimParam[1])                        
                        for tt in range(ntrials):
                            signal = swps_new[stmIdx[tt]].signal[:nsamples+100, cc]
#                            print(signal.size)
                            if tt == 0:
                                Ref = signal[:sampleIdx]
                                peakIdxRef = np.argmax(np.correlate(Ref-np.mean(Ref),Ref-np.mean(Ref),'full')[(sampleIdx-5):(sampleIdx+5)])                           
                                signal_arrays[cc, ff, dd, ii, jj, :, tt] = signal[: nsamples]
                            if tt >= 1:
                                Temp = signal[:sampleIdx]
                                peakIdxtemp = np.argmax(np.correlate((Ref-np.mean(Ref)),(Temp-np.mean(Temp)),'full')[(sampleIdx-5):(sampleIdx+5)])
                                peakdiff = peakIdxRef-peakIdxtemp
                                if peakdiff < 0:
                                    signal = np.concatenate((np.zeros((np.abs(peakdiff)), dtype = 'float32'), signal))
                                    signal_arrays[cc, ff, dd, ii, jj, :, tt] = signal[: nsamples]
                                elif peakdiff >= 0:
                                    signal = signal[peakdiff: peakdiff+nsamples]
                                    signal_arrays[cc, ff, dd, ii, jj, :, tt] = signal[: nsamples]
    # plot error mark 
    plt.figure(figsize=(10,15))
    plt.xticks(np.arange(0, 32, 1))
    plt.subplot(1,2,1)
    plt.title('900pps')
    y = 0
    for dd in range(3):
        for ii in range(3):
            for jj in range(3):
                y = y+1
                for cc in range(32):
                    if ErrorMark_array[cc, 0, dd, ii, jj] == 0:                        
                        plt.plot(cc+1,y, 'ko')
                    else:
                        plt.plot(cc+1,y, 'ro')
    plt.subplot(1,2,2)
    plt.title('4500pps')
    y = 0
    for dd in range(3):
        for ii in range(3):
            for jj in range(3):
                y = y+1
                for cc in range(32):
                    if ErrorMark_array[cc, 1, dd, ii, jj] == 0:                        
                        plt.plot(cc+1,y, 'ko')
                    else:
                        plt.plot(cc+1,y, 'ro')
    plt.savefig(results_path+fname)
    plt.close()
    # save data
    if nchans == 32: 
        np.save(results_path+fname+'_OriginSigArray.npy',signal_arrays)
        np.save(results_path+fname+'_Stimulus.npy',stm)
    if nchans == 64: 
        np.save(results_path+fname+'_IC_OriginSigArray.npy',signal_arrays[:32, :, :, :, :, :, :])
        np.save(results_path+fname+'_AC1_OriginSigArray.npy',signal_arrays[32:64, :, :, :, :, :, :])
        np.save(results_path+fname+'_Stimulus.npy',stm)
    if nchans == 96:
        np.save(results_path+fname+'_IC_OriginSigArray.npy',signal_arrays[:32, :, :, :, :, :, :])
        np.save(results_path+fname+'_AC1_OriginSigArray.npy',signal_arrays[32:64, :, :, :, :, :, :])
        np.save(results_path+fname+'_AC2_OriginSigArray.npy',signal_arrays[64:96, :, :, :, :, :, :])
        np.save(results_path+fname+'_Stimulus.npy',stm)

    