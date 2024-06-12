# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:33:02 2024

@author: shiyi

This module design for ENVvsPT ephys analysis
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, csd, windows, welch, get_window, resample, lfilter
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from tkinter import Tk
from tkinter.filedialog import askdirectory
import os
from scipy import stats

Fs = 24414.0625
stiDur = [0.01, 0.05, 0.2]
stiRate = [900, 4500]
stiITD = [-0.1, 0, 0.1]
nChannel = 32
nRate = 2
nITD = 3
nDur = 3

def MaxIdx(dd, x):
    idx_max = int(stiDur[dd]*Fs)+x
    return idx_max  

def ask_directory():
    root = Tk()
    sig_path = askdirectory(title='Select Folder')
    root.withdraw()
    return sig_path

def get_names(sig_path, file_tile):
    file_names = os.listdir(sig_path)
    ori_names = [file_name for file_name in file_names if all([x in file_name for x in [file_tile]])]
    return ori_names

def load_data(name, path):
    sig = np.load(path+'/'+name, allow_pickle = True)
    return sig

def load_original(ori_name, sig_path):
    ori_array = load_data(ori_name, sig_path)
    return ori_array

    

class EphysParameter():
    
    def __init__(self):
        self.Fs = 24414.0625
        self.stiDur = [0.01, 0.05, 0.2]
        self.stiRate = [900, 4500]
        self.stiITD = [-0.1, 0, 0.1]
        self.nRate = 2
        self.nDur = 3
        self.nITD = 3
        self.nChannel = 32
        '''
        AMUA Filter
        '''
        self.lowpass = 6000
        self.bandpassA = 300
        self.bandpassB = 6000
        self.lowpassB = 200
        self.Notchw0 = 50
        self.NotchQ = 30
        self.Fs_downsample = 2000
        self.AMUAFilterCoeffs()
        
        
    def AMUAFilterCoeffs(self):
        nyq = 0.5*Fs
        self.bBand, self.aBand = butter(2,(self.bandpassA/nyq, self.bandpassB/nyq),'bandpass')
        self.bLow,self.aLow = butter(2,(self.lowpass/nyq),'lowpass')
        self.bNotch, self.aNotch = iirnotch(self.Notchw0, self.NotchQ, Fs)
        self.Wn = 2*200/self.Fs_downsample
        self.bLowB, self.aLowB = butter(2, self.Wn, 'lowpass')    
        
    
class ArtifactRejection(EphysParameter):

    # Artfact Rejection

    def __init__(self, original_sig):
        super().__init__()
        self.ori_array = original_sig
        self.nSamples = self.ori_array.shape[-2]
        self.nTrials = self.ori_array.shape[-1]
        self.kaiserBeta = 5
        self.kaiserN = 256
        self.cc = None
        self.ff = None
        self.dd = None
        self.ii = None
        self.jj = None
        self.ARR_trace_array = None
        self.ARR_array = None
        
    '''
    Below methods reject artifact from original data    
    '''
    def reject_artifact(self, cc, ff, dd, ii, jj):
        self.calc_avg(self.ori_array[cc, ff, dd, ii, jj, :, :])
        self.subtract_avg(cc, ff, dd, ii, jj)
        return self.clean_array[cc, ff, dd, ii, jj, :, :]
    
    def clean_array_ready(self):
        self.clean_array = np.zeros((self.nChannel, self.nRate, self.nDur, self.nITD, self.nITD, self.nSamples, self.nTrials) ,dtype = 'float32')
    
    def subtract_avg(self, cc, ff, dd, ii, jj):
        self.clean_array[cc, ff, dd, ii, jj, :, :] = self.ori_array[cc, ff, dd, ii, jj, :, :]-self.template_array
    
    def calc_avg(self, sig):
        # averge out the trials
        template = np.mean(sig, -1)
        self.template_array = np.array([template]*self.nTrials).T
        
    def save(self, ori_name, sig_path):
        # save clean sig
        name = ori_name[:-19]
        # clean_name = name+'_CleanSigArray.npy'
        # np.save(sig_path+clean_name, self.clean_array)
        # save ARR trace
        ARR_trace_name = name+'_CleanSig_ARRTrace.npy'
        np.save(sig_path+'/'+ARR_trace_name, self.ARR_trace_array)
        # save ARR freq
        ARR_freq_name = name+'_CleanSig_ARRfreq.npy'
        np.save(sig_path+'/'+ARR_freq_name, self.f)
        # save ARR avg
        ARR_avg_name = name+'_CleanSig_ARRavg.npy'
        np.save(sig_path+'/'+ARR_avg_name, self.ARR)
        
        
        
    '''
    Below methods calculate ARR
    Evaluate artifact rejection quality
    '''
        
    def ARR_array_ready(self):
        self.ARR_array = np.zeros((self.nChannel, self.nRate, self.nDur, self.nITD, self.nITD) ,dtype = 'float32')
        
    def ARR_trace_array_ready(self):
        ARR_nsamples = self.ARR.shape[0]
        self.ARR_trace_array = np.zeros((self.nChannel, self.nRate, self.nDur, self.nITD, self.nITD, ARR_nsamples), dtype = 'float32')
                
    def calc_ARR(self, cc, ff, dd, ii, jj):
        # ARR = SNR_post/SNR_pre
        # SNR = PSD-CSD/CSD
        self.kaiserWin()
        self.f, self.SNR_post = self.calc_SNR(self.ori_array[cc, ff, dd, ii, jj, :, 0], self.ori_array[cc, ff, dd, ii, jj, :, 1])
        self.f, self.SNR_pre = self.calc_SNR(self.clean_array[cc, ff, dd, ii, jj, :, 0], self.clean_array[cc, ff, dd, ii, jj, :, 1])
        self.ARR = self.SNR_post/self.SNR_pre
        if self.ARR_trace_array is None:
            self.ARR_trace_array_ready()
        self.ARR_trace_array[cc, ff, dd, ii, jj, :] = self.ARR        

    def kaiserWin(self):
        self.kaiser = get_window(('kaiser', self.kaiserBeta), self.kaiserN)
    
    def calc_SNR(self, sig1, sig2):
        f_csd, sig_csd = self.calc_CSD(sig1, sig2)
        f_psd, sig_psd = self.calc_PSD(sig1)
        sig_snr = (sig_psd-sig_csd)/sig_csd
        return f_csd, sig_snr
    
    def calc_CSD(self, sig1, sig2):
        f_csd, sig_csd = csd(sig1, sig2, Fs, window = self.kaiser, nperseg = len(self.kaiser))
        return f_csd, sig_csd
        
    def calc_PSD(self, sig1):
        f_psd, sig_psd = welch(sig1, Fs, window = self.kaiser, nperseg = len(self.kaiser))
        return f_psd, sig_psd
    
    '''
    Method plot ARR against Freq
    Needs update
    '''
    def ARR_plot(self):
        ax = plt.gca()
        ax.plot(self.f, self.ARR)
        return(ax)
    
    '''
    Method calculate ARR average against Harmonic freq
    '''
    def calc_ARRmean(self, cc, ff, dd, ii, jj):
        if ff != self.ff:           
            self.harmonic_check(ff)
            self.ff = ff
        self.avg_ARR(cc, ff, dd, ii, jj)
    
    def harmonic_check(self, ff):
        freq_del = self.f[1]
        freq_up = self.f[-1]
        F0 = self.stiRate[ff]
        freqs = np.arange(0, freq_up, F0)[1:]
        self.HarmonicIdx = np.asarray((np.around(freqs/freq_del)), dtype = 'int')
        
    def avg_ARR(self, cc, ff, dd, ii, jj):
        HarmonicARR = [self.ARR[n] for n in self.HarmonicIdx]
        AverageARR = np.mean(HarmonicARR)
        ARRindB = round(20*np.log10(AverageARR), 2)
        self.ARR_array[cc, ff, dd, ii, jj] = ARRindB
        


class AMUACalc(EphysParameter):
    def __init__(self):
        super().__init__()
        self.cleansig = None
        self.response_start = 0.005
        self.response_end = 0.055
        self.baseline_start = 0.45
        self.AMUA_array = None
        self.AMUA_resp_avg_array = None
        self.AMUA_base_avg_array = None
        self.nsamples = None
        self.ntrials = None
        self.wilcoxon_array = None
    
    def AMUA_array_ready(self):
        t = self.nsamples/self.Fs
        self.nsamples_down = int(self.Fs_downsample*t)
        self.AMUA_array = np.zeros((self.nChannel, self.nRate, self.nDur, self.nITD, self.nITD, self.nsamples_down, self.ntrials))
        self.AMUA_filter_array = np.zeros((self.nChannel, self.nRate, self.nDur, self.nITD, self.nITD, self.nsamples_down, self.ntrials))
        
    def calc_AMUA(self, cleansig, cc, ff, dd, ii, jj, padLen=300, wilcoxon = True):
        self.cleansig = cleansig
        '''
        generate AMUA_array
        '''
        if self.AMUA_array is None:
            self.nsamples = self.cleansig.shape[-2]
            self.ntrials = self.cleansig.shape[-1]
            self.AMUA_array_ready()
        '''
        Calculate AMUA
        '''
        insig = filtfilt(self.bNotch, self.aNotch, self.cleansig, axis=0, padlen=padLen)
        insig = np.flip(insig)
        insig = filtfilt(self.bBand,self.aBand, insig, axis=0, padlen=padLen)
        insig = np.abs(insig)
        insig = filtfilt(self.bLow,self.aLow,insig,axis=0, padlen=padLen)
        insig = np.flip(insig)
        insig = resample(insig,self.nsamples_down)
        self.AMUA_array[cc, ff, dd, ii, jj, :, :] = insig
        '''
        Filter AMUA by low pass
        '''
        self.AMUA_filter(insig, cc, ff, dd, ii, jj)
        if wilcoxon:
            self.AMUA_avg(cc, ff, dd, ii, jj)
    
    def AMUA_filter(self, insig, cc, ff, dd, ii, jj):
        insig = filtfilt(self.bLowB, self.aLowB, insig, axis=0, padlen=100)
        self.AMUA_filter_array[cc, ff, dd, ii, jj, :, :] = insig
        
    def AMUA_avg(self, cc, ff, dd, ii, jj, resp_start = 0.005, resp_end = 0.055, base_start = 0.45, base_end = 0.5):
        insig = self.AMUA_filter_array[cc, ff, dd, ii, jj, :, :]
        resp = np.mean(insig[int(self.Fs_downsample*resp_start):int(self.Fs_downsample*resp_end), :], 0)
        base = np.mean(insig[int(self.Fs_downsample*base_start):int(self.Fs_downsample*base_end), :], 0)
        if  self.AMUA_resp_avg_array is None:
            self.AMUA_avg_array_ready()
        self.AMUA_resp_avg_array[ff, dd, ii, jj, :] = resp
        self.AMUA_base_avg_array[ff, dd, ii, jj, :] = base
            
    def AMUA_avg_array_ready(self):
        self.AMUA_resp_avg_array = np.zeros((self.nRate, self.nDur, self.nITD, self.nITD, self.ntrials))
        self.AMUA_base_avg_array = np.zeros((self.nRate, self.nDur, self.nITD, self.nITD, self.ntrials))
        
    def AMUA_wilcoxon(self, cc):
        if self.wilcoxon_array is None:
            self.wilcoxon_array_ready()
        self.wilcoxon_array[cc, :, 0] = np.reshape(self.AMUA_resp_avg_array, (1, self.nRate*self.nDur*self.nITD*self.nITD*self.ntrials))[0]
        self.wilcoxon_array[cc, :, 1] = np.reshape(self.AMUA_base_avg_array, (1, self.nRate*self.nDur*self.nITD*self.nITD*self.ntrials))[0]
        self.wilcoxon_results[cc, :] = stats.wilcoxon(self.wilcoxon_array[cc, :, 0], self.wilcoxon_array[cc, :, 1])
    
    def wilcoxon_array_ready(self):
        self.wilcoxon_array = np.zeros((self.nChannel, self.nRate*self.nDur*self.nITD*self.nITD*self.ntrials, 2))
        self.wilcoxon_results = np.zeros((self.nChannel,2))
        
    def save(self, ori_name, sig_path):
        name = name = ori_name[:-19]
        
        # AMUA_name = name+'_AMUA.npy'
        # np.save(sig_path+AMUA_name, self.AMUA_array)
        
        AMUA_filtered_name = name+'_AMUAfiltered.npy'
        np.save(sig_path+'/'+AMUA_filtered_name, self.AMUA_filter_array)
        
        AMUA_avg_name = name+'_AMUAavg.npy'
        np.save(sig_path+'/'+AMUA_avg_name, self.wilcoxon_array)
        
        AMUA_wilcoxon_name = name+'_AMUA_wilcoxon.npy'
        np.save(sig_path+'/'+AMUA_wilcoxon_name, self.wilcoxon_results)
        
        
        
        

        
    

    
    

# class ArtifactRemoveQuality(EphysParameter):
    
#     def __init__(self, original_sig, clean_sig): # sig in channel
#         super().__init__()
#         self.kaiserBeta = 5
#         self.kaiserN = 256
        
#     def calc_ARR(self):
#         self.clean_SNR = self.calc_SNR()
#         self.ori_SNR = self.calc_SNR()
#         self.ARR = self.clean_SNR/self.ori_SNR        
    
#     def kaiserWin(self):
#         kaiser_window = get_window(('kaiser', self.kaiserBeta), self.kaiserN)
#         return kaiser_window
    
#     def CalcPSD(self, sig):
#         win = self.kaiserWin()
#         f_psd, sig_psd = welch(sig, Fs, window = win, nperseg = len(win))
#         return f_psd, sig_psd
    
#     def CalcCSD(self, sig1, sig2):
#         win = self.kaiserWin()
#         f_csd, sig_csd = csd(sig1, sig2, Fs, window = win, nperseg = len(win))
#         return f_csd, sig_csd    
    
#     def CalcSNR(self, sig1, sig2):
#         f_csd, sig_csd = self.CalcCSD(sig1, sig2)
#         f_psd, sig_psd = self.CalcPSD(sig1)
#         sig_snr = (sig_psd-sig_csd)/sig_csd
#         return f_psd, sig_snr
    
#     def CalcARR(self, sig1, sig2):
#         ARR = sig1/sig2        
#         return np.abs(ARR)
    
    # def GetHarmonicIdx(self, del_f, freq_up, F0):
    #     i = np.arange(0, freq_up, F0)[1:]
    #     idx = np.asarray((np.around(i/del_f)), dtype = 'int')
    #     self.HarmonicIdx = idx
    
    # def CheckHarmonicARR(self, freq, sig, F0):
    #     del_f = freq[1]
    #     freq_up = freq[-1]
    #     self.GetHarmonicIdx(del_f, freq_up, F0)
    #     self.HarmonicARR = [sig[n] for n in self.HarmonicIdx]
    #     # self.ARRmean()
    #     # return self.ARRAvg
    
    # def ARRmean(self, freq, sig, F0):
    #     self.CheckHarmonicARR(freq, sig, F0)
    #     # self.GetHarmonicIdx(del_f, freq_up, F0)
    #     self.ARRAvg = np.mean(self.HarmonicARR)
    #     ARRindB = round(20*np.log10(self.ARRAvg), 2)        
    #     return ARRindB    

# class GetName(AnalysisEphys):
#     def __init__(self, name, sig_path):
#         self.ori_name = name
#         self.sig_path = sig_path
#         self.clean_name = None
#         self.getCleanName()
#         self.AMUA_name = None
#         self.getAMUAName()     
        
#     def getCleanName(self):
#         self.clean_name = self.ori_name[:-4]+'_CleanSig_Mean.npy'        
    
#     def getAMUAName(self):
#         self.AMUA_name = self.ori_name[:-4]+'_AMUA_Mean.npy' 
        
#     def getARRName(self):
#         self.ARR_name = self.ori_name[:-4]+'_ARRindB_Mean.npy'
        
#     def getAMUACleanfig(self, cc):
#         self.AMUAClean_fig = self.ori_name[:-4]+'_AMUA_ch'+str(cc)
        
#     def loadClean(self):
#         self.clean_sig = np.load(self.sig_path+self.clean_name, allow_pickle = True)
    
#     def loadAMUA(self):
#         self.AMUA_sig = np.load(self.sig_path+self.AMUA_name, allow_pickle = True)
        
#     def loadARR(self):
#         self.ARR_sig = np.load(self.sig_path+self.ARR_name, allow_pickle = True)
    
# class AMUACalc(GetName):
    
#     def __init__(self, sig_name, sig_path):
#         super().__init__(sig_name, sig_path)
#         self.fs = 24414.0625
#         self.response_start = 0.005
#         self.response_end = 0.055
#         self.baseline_start = 0.45
#         self.lowpass = 6000
#         self.bandpassA = 300
#         self.bandpassB = 6000
#         self.lowpassB = 200
#         self.padLen = 300
#         self.Notchw0 = 50
#         self.NotchQ = 30
#         self.Fs_downsample = 2000
#         self.nblank = 5
#         # self.sig = sig
#         self.t = None
#         self.nsamples_down = None
#         self.ntrials = None
#         self.PredicSize()
#         self.bBand = None
#         self.aBand = None
#         self.bLow = None
#         self.aLow = None
#         self.bLowB = None
#         self.aLowB = None
#         self.bNotch = None
#         self.aNotch = None
#         self.AMUAFilterCoeffs()
#         self.amua_array = None
#         self.CreatSelectionArray()
#     # def MaxIdxAMUA(self, dd, x):
#     #     idx_max = int(stiDur[dd]*self.Fs_downsample)+x
#     #     return idx_max
    
#     def AMUAFilterCoeffs(self):
#         nyq = 0.5*Fs
#         self.bBand, self.aBand = butter(2,(self.bandpassA/nyq, self.bandpassB/nyq),'bandpass')
#         self.bLow,self.aLow = butter(2,(self.lowpass/nyq),'lowpass')
#         self.bNotch, self.aNotch = iirnotch(self.Notchw0, self.NotchQ, Fs)
#         self.Wn = 2*200/self.Fs_downsample
#         self.bLowB, self.aLowB = butter(2, self.Wn, 'lowpass')               
    
#     def calcAMUA(self, cc, ff, dd, ii, jj, padLen=300):
#         '''
#         cosidering some clean signal padding 1-3 zeros at begining to align up
#           in case the padding will affect the AMUA and frequency domain results
#         fllowing is checking code
#         '''
#         # coefs= self.AMUAFilterCoeffs()
#         # bpCoefs=coefs[0]
#         # lpCoefs=coefs[1]
#         # NotchCoefs = coefs[2]
#         insig = self.clean_sig[cc, ff, dd, ii, jj, self.nblank:, :]
#         insig = filtfilt(self.bNotch, self.aNotch, insig, axis=0, padlen=padLen)
#         insig = np.flip(insig)
#         insig=filtfilt(self.bBand,self.aBand, insig, axis=0, padlen=padLen)
#         insig=np.abs(insig)
#         insig=filtfilt(self.bLow,self.aLow,insig,axis=0, padlen=padLen)
#         insig = np.flip(insig)
#         self.amua_array[cc, ff, dd, ii, jj, :, :] = self.resampleAMUA(insig)
#         self.temp = self.resampleAMUA(insig)
#         # return signal
        
#     def resampleAMUA(self, insig):
#         # signal = resample_poly(insig, Fs_downsample, int(fs), axis=0)
#         signal=resample(insig,self.nsamples_down)
#         return signal
    
#     # def creatArray(self, dim):
#     #     if dim == 1:
#     #         array = np.zeros((self.nchannel))
        
    
#     def CreatAMUAarray(self):
#         self.amua_array = np.zeros((nChannel, nRate, nDur, nITD, nITD, self.nsamples_down, self.ntrials))
    
#     def CreatArrayincc(self):
#         self.selection_array = np.zeros((nChannel))
    
#     def CreatSelectionArray(self):
#         self.resp_array = np.zeros((nRate, nDur, nITD, nITD, self.ntrials))
#         self.basel_array = np.zeros((nRate, nDur, nITD, nITD, self.ntrials))
            
#     def PredicSize(self):
#         self.loadClean()
#         self.ntrials = self.clean_sig.shape[-1]
#         self.t = (self.clean_sig.shape[-2]-self.nblank)/Fs        
#         self.nsamples_down = int(self.Fs_downsample*self.t)
        
#     def filtAMUA(self):
#         insig = filtfilt(self.bLowB, self.aLowB, self.amua_temp, axis=0, padlen=100)
#         self.amua_temp = insig
    
#     def meanAMUA(self, start, end):
#         if end == None:
#             insig = np.mean(self.amua_temp[start:], 0)
#         else:
#             insig = np.mean(self.amua_temp[start:end], 0)   
#         return insig
        
#     def prepRespBaseline(self, cc, ff, dd, ii, jj):
#         self.amua_temp = self.temp
#         self.filtAMUA()
#         self.resp_array[ff, dd, ii, jj] = self.meanAMUA(int(self.Fs_downsample*self.response_start), int(self.Fs_downsample*self.response_end))
#         self.basel_array[ff, dd, ii, jj] = self.meanAMUA(int(self.Fs_downsample*self.baseline_start), None)
        
#     def wilcoxonAMUA(self, cc):
#         response = np.reshape(self.resp_array, (1, nRate*nDur*nITD*nITD*self.ntrials))[0]
#         baseline = np.reshape(self.basel_array, (1, nRate*nDur*nITD*nITD*self.ntrials))[0]
#         results = stats.wilcoxon(response, baseline)
#         if results[1] < 0.05:
#             self.selection_array[cc] = 1
            
            
# class Plot(GetName):
    
#     def __init__(self, sig_name, sig_path):
#         super().__init__(sig_name, sig_path)
#         self.Fs_downsample = 2000
#         # self.cc = cc
#         self.t = 0.2 #s
#         self.x = None
#         self.nsamples = None
#         self.AMUA_x = self.CalcXaxis(self.Fs_downsample)
#         self.clean_x = self.CalcXaxis(Fs)
#         self.Wn = None
#         self.nyq = Fs/2
#         # self.Creatfig()
        
#     def Creatfig36(self, cc):
#         self.cc = cc
#         self.fig = plt.figure(figsize=(20, 10))
#         # self.fig.set_tight_layout(True)
#         self.gs = GridSpec(nrows = 3, ncols = 6, figure = self.fig)
#         self.fig.suptitle(self.ori_name[:-4]+'_Ch'+str(self.cc), fontsize=16)
    
#     def CalcXaxis(self, fs):
#         x = np.arange(0, self.t, 1/fs)
#         return x
        
#     def plotAMUAClean(self, cc, ff, dd, ii, jj):
#         self.ff = ff; self.dd = dd; self.ii = ii; self.jj = jj
#         self.PrepareAMUA2Plot(cc, ff, dd, ii, jj)
#         self.PrepareClean2Plot(cc, ff, dd, ii, jj)
#         ax1 = self.fig.add_subplot(self.gs[ii, jj])
#         ax1.set_title('PT_ITD: '+str(stiITD[ii])+' ENV_ITD: '+str(stiITD[jj]))
#         ax1.plot(self.AMUA_x, self.AMUA_temp[:len(self.AMUA_x)]*1000000)
#         ax1.set_ylim(2, 35)
#         ax2 = self.fig.add_subplot(self.gs[ii, jj+3])
#         ax2.set_title('PT_ITD: '+str(stiITD[ii])+' ENV_ITD: '+str(stiITD[jj]))
#         ax2.plot(self.clean_x, self.Clean_temp[:len(self.clean_x)]*1000000)
    
#     def PrepareAMUA2Plot(self, cc, ff, dd, ii, jj):
#         # prepare AMUA ready to plot, save in self.temp
#         self.AMUA_temp = self.AMUA_sig[cc, ff, dd, ii, jj, :, :]
#         self.Clean_temp = self.clean_sig[cc, ff, dd, ii, jj, :, :]
#         self.AMUAFilt(self.AMUA_temp) # 200Hz lowpass filter
#         self.AMUAMean(self.AMUA_temp) # average over trials
         
#     def AMUAFilt(self, insig):
#         self.Wn = 2*200/self.Fs_downsample
#         bLow,aLow = butter(2, self.Wn, 'lowpass')
#         insig = filtfilt(bLow, aLow, insig, axis=0, padlen=100)
#         self.AMUA_temp = insig
    
#     def AMUAMean(self, insig):
#         insig = np.mean(insig, -1)
#         self.AMUA_temp = insig
    
#     def PrepareClean2Plot(self, cc, ff, dd, ii, jj):
#         self.Clean_temp = self.clean_sig[cc, ff, dd, ii, jj, 1:, :]
#         self.CleanFilt(self.Clean_temp)
        
#     def CleanFilt(self, insig):
#         bhigh2,ahigh2 = butter(2, 3000/self.nyq, 'highpass')
#         insig = filtfilt(bhigh2,ahigh2, insig, axis=0, padlen=100)
#         self.Clean_temp = insig
        
#     def figSave(self, savepath):
#         self.getAMUACleanfig(self.cc)
#         self.fig.savefig(savepath+self.AMUAClean_fig)
#         plt.close(self.fig)
        
# class ArtifactRemoveQuality:
    
#     def __init__(self, Fs = 24414.0625, kaiserBeta = 5, kaiserN = 256):
#         self.Fs = Fs
#         self.kaiserBeta = kaiserBeta
#         self.kaiserN = kaiserN
        
#     def kaiserWin(self):
#         kaiser_window = get_window(('kaiser', self.kaiserBeta), self.kaiserN)
#         return kaiser_window
    
#     def CalcPSD(self, sig):
#         win = self.kaiserWin()
#         f_psd, sig_psd = welch(sig, Fs, window = win, nperseg = len(win))
#         return f_psd, sig_psd
    
#     def CalcCSD(self, sig1, sig2):
#         win = self.kaiserWin()
#         f_csd, sig_csd = csd(sig1, sig2, Fs, window = win, nperseg = len(win))
#         return f_csd, sig_csd    
    
#     def CalcSNR(self, sig1, sig2):
#         f_csd, sig_csd = self.CalcCSD(sig1, sig2)
#         f_psd, sig_psd = self.CalcPSD(sig1)
#         sig_snr = (sig_psd-sig_csd)/sig_csd
#         return f_psd, sig_snr
    
#     def CalcARR(self, sig1, sig2):
#         ARR = sig1/sig2        
#         return np.abs(ARR)
    
#     def GetHarmonicIdx(self, del_f, freq_up, F0):
#         i = np.arange(0, freq_up, F0)[1:]
#         idx = np.asarray((np.around(i/del_f)), dtype = 'int')
#         self.HarmonicIdx = idx
    
#     def CheckHarmonicARR(self, freq, sig, F0):
#         del_f = freq[1]
#         freq_up = freq[-1]
#         self.GetHarmonicIdx(del_f, freq_up, F0)
#         self.HarmonicARR = [sig[n] for n in self.HarmonicIdx]
#         # self.ARRmean()
#         # return self.ARRAvg
    
#     def ARRmean(self, freq, sig, F0):
#         self.CheckHarmonicARR(freq, sig, F0)
#         # self.GetHarmonicIdx(del_f, freq_up, F0)
#         self.ARRAvg = np.mean(self.HarmonicARR)
#         ARRindB = round(20*np.log10(self.ARRAvg), 2)        
#         return ARRindB
    
        
        

        
        
    

    

        
        
        
        
        
        
        
        
        