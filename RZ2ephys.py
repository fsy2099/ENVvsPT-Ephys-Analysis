# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 12:54:10 2018

@author: Jan
"""

import numpy as np
#import matplotlib.pyplot as plt
#import time
from io import IOBase
import struct
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt, decimate, iirnotch
import sys
import time
import copy
import pandas as pd
from matplotlib import pyplot as plt
# import gc
import psyPhysConfig as config
verbose=False
#%%

#class MchanDataStream:
#    def __init__(self, source):
#        self.nChan=nChan
#        self.sweeplen=1000

## for the continuous recording we use a timer thread whcih is copied from
# https://stackoverflow.com/questions/3393612/run-certain-code-every-n-seconds

from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.adjustment   = 0
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.tic=0
        self.stopped=False
        #self.start()

    def _run(self):
        if self.stopped:
            return
        self.function(*self.args, **self.kwargs)
        self.adjustment=time.time()-self.tic
        self.is_running = False
        self.start()

    def start(self):
        self.stopped=False
        if not self.is_running:
            nextInterval=max(self.interval-self.adjustment,0.01)
            self._timer = Timer(nextInterval, self._run)
            self.tic=time.time()
            self._timer.start()
            self.is_running = True

    def stop(self):
        self.stopped=True
        if not self._timer == None:
            self._timer.cancel()
        self.is_running = False
        self.adjustment   = 0
        
        
#%%
class Virtual_Cont_Recorder:
# virtual class of continuous ephys recorders. 
# THis virtual recorder will produce simulated data.
# From this we derive hardware specific classes which record actual data, e.g. the one below.
# Continuous recorders organise the data in sweeps that go from one trigger event to the 
#   next. Sweeps are therefore variable in length.
    def __init__(self, ephysFileName=None):
        self.running=False
        self.sweepLen=1.0 
        self.sweeps=[]
        self.sweepToPlot=None
        self.sigBuf=np.array([]).astype('float32')
        self.lastIdx=0
        self.lastError=''
        self.totalUploaded=0
        self.sweepToProcess=None
        self.ephysFileName= ephysFileName
        self.ephysFileHandle= None    
        self.sweepEnd=0 
        self.uploadBufferTimer=None
        self.processBufferTimer=None
        self.timestampBuffer=[time.time()]
        self.spikeFilter = None
        self.sampleRate=25000
        # work out how many channels there are
        self.nChan=16
        self.displayLastSavedSweepCols=0
        self.figure=None
        self.lines=[]
        
    def nyquist(self):
        return self.sampleRate/2
            
    def spikeFilterCoeffs(self):
        b,a = butter(4,(300/self.nyquist(), 3000/self.nyquist()),'bandpass')
        return [b,a]
        
    def uploadData(self):
        # Fill recording buffer with new data. Return True if a new sweep was completed, false otherwise
        if not self.uploadBufferTimer is None:
            self.uploadBufferTimer.stop() # stop timer to prevent collisions
        # this is called repeatedly by timer to upload data recorded by the ahrdware to buffers
        try:
            # for the virtual recorder, make noise buffers to simulate recorder input
            now=time.time()
            if not hasattr(self,'lastUploadAt'):
                self.lastUploadAt=self.recordingStartedAt
                self.sweepStart=0
                self.sweepEnd=1e13
            timeSinceLastUpload=now-self.lastUploadAt
            self.lastUploadAt=time.time()
            sweepEnd=int((now-self.recordingStartedAt)*self.sampleRate)
            nextChunkSize=int(timeSinceLastUpload*self.sampleRate)
            self.totalUploaded+=nextChunkSize
            sig=np.random.random(self.nChan*nextChunkSize).astype('float32')
                
            self.sigBuf=np.append(self.sigBuf,sig,axis=0)
            # check if a sweep has ended.
            if sweepEnd > self.sweepEnd+1:
                self.processNewSweep(self.sweepStart,self.sweepEnd)
                self.sweepStart=self.sweepEnd
                self.sweepEnd=1e13
                return True
            else:
                return False
            self.uploadBufferTimer.start # restart timer               
        except BaseException as e:
            # make sure system timer is halted if error occurs
            self.stop()
            self.lastError=str(e)
            raise     
            
    def trigger(self):
        # a trigger marks that a sweep has ended.
        now=time.time()
        self.sweepEnd=int((now-self.recordingStartedAt)*self.sampleRate)
        self.timestampBuffer.append(now)
        
    def processNewSweep(self,sweepStart,sweepEnd):
        # if verbose: 
        #     print('processNewSweep()')
        sweepLen=sweepEnd-sweepStart
        nSamplesExpected=sweepLen*self.nChan
        nSamplesGot=int(np.floor(len(self.sigBuf)/self.nChan)*self.nChan)
        if nSamplesExpected > nSamplesGot:
            print()
            print('*** ERROR *** ERROR *** ERROR *** ERROR ')
            print('*** processNewSweep expected: {} got:{}. '.format(nSamplesExpected,nSamplesGot)); sys.stdout.flush()
            print()
            self.sweepEnd=sweepEnd
            config.status=="abort"
            raise Exception('Error processing data buffers')
            return
        if len(self.timestampBuffer) == 0:
            tStamp=0.0
        else:
            tStamp=self.timestampBuffer.pop(0)
        newSweep=MCsweep(self.sigBuf[:nSamplesExpected],sweepLen,self.sampleRate,tStamp)
        self.sweeps.append(newSweep)
        oldLen=len(self.sigBuf)
        self.sigBuf=self.sigBuf[nSamplesExpected:]
        self.sweepEnd=sweepEnd
        print('--- Added {} samples to new sweep with time stamp {:.1f}. {} sweeps in RAM'.format(nSamplesExpected, tStamp, len(self.sweeps)))
        print()
        print('--- sigBuf length went *** from {:,} to {:,} ***.'.format(oldLen, len(self.sigBuf)))
        print()
        
    def samplesInSigBuf(self):
        return int(len(self.sigBuf)/self.nChan)
        
    def secondsInSigBuf(self):
        return self.samplesInSigBuf/self.sampleRate
    
    def sweepAcquired(self):
        return self.secondsInSigBuf() >= self.sweepLen
        
    def runningFor(self):
        return time.time()-self.recordingStartedAt
    
    def start(self):
        self.running=True
        self.lastIdx=0
        self.sigBuf=np.array([]).astype('float32')
        self.sweepStart=0
        self.sweepEnd=0    
        self.totalUploaded=0
        self.timestampBuffer=[time.time()]
        self.uploadBufferTimer=RepeatedTimer(self.sweepLen,self.uploadData)
        self.recordingStartedAt=time.time()
        self.uploadBufferTimer.start()
        print('recorder started.')
        sys.stdout.flush()
    
    def stop(self):
        self.running=False
        if self.uploadBufferTimer.is_running:
            self.uploadBufferTimer.stop()
            # collect the last sweep
            self.uploadData()
            self.processNewSweep(0,self.samplesInSigBuf())
        print('recorder stopped.')
        sys.stdout.flush()

    def plotSweep(self):
        try:
            if self.sweepToPlot is None:
                # print('plotSweep(): nothing to plot!')
                return
            if self.figure is None:
                # allocating new figure
                self.figure=plt.figure()
            # print('switching interactive mode off '); sys.stdout.flush()
            # plt.ioff()
            if not self.spikeFilter is None:
                self.sweepToPlot=self.sweepToPlot.IIRfilter(self.spikeFilter)
            self.lines=self.sweepToPlot.plotSig(nCol=self.displayLastSavedSweepCols, pltHandle=self.figure, lineObjects=self.lines)
            # print('called plotSig()'); sys.stdout.flush()
            # finally, make sure we are done plotting
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
        except Exception as e:
            print('NOTE: plotSweep() failed !', str(e))
        # then delete the sweep    
        self.sweepToPlot=None
        # gc.collect()
        # print('switching interactive mode back on'); sys.stdout.flush()
        # plt.ion()

    def saveAcquiredSweeps(self):
        # if there is a data file, save freshly acquired sweeps
        if len(self.sweeps)<=0:
            return # nothing to save, so exit
        if self.displayLastSavedSweepCols > 0:
            if self.sweepToPlot is None:
                self.sweepToPlot=copy.deepcopy(self.sweeps[-1])
                # self.plotThread=Timer(0.1, self.plotSweep)
                # self.plotThread.start()
        if self.ephysFileHandle is None:
            if not self.ephysFileName is None:
            # we have a file name but no file handle. 
            #  -> open file handle and try again
                print('opening data file '+self.ephysFileName)
                self.ephysFileHandle = open(self.ephysFileName,'ab')
            else:
                # in some old versions ephysFileName may be called ephysFile.
                # if that's the case, cast and retry
                if  hasattr(self,'ephysFile'):
                    if self.ephysFile is IOBase:
                        self.ephysFileHandle=self.ephysFile
                    else:
                        self.ephysFileName=self.ephysFile
                        print('opening data file '+self.ephysFileName)
                        self.ephysFileHandle = open(self.ephysFileName,'ab')
        if not self.ephysFileHandle is None:
            print('Saving {} sweep(s)'.format(len(self.sweeps)))
            while len(self.sweeps)>0:
                self.sweepToProcess=self.sweeps.pop(0)
                self.sweepToProcess.saveSweep(self.ephysFileHandle)
                # we always keep the last few sweeps in memory a bit longer for plotting

            
#%%     test Virtual Recorder
#r=Virtual_Cont_Recorder()
#r.start()
#time.sleep(0.1)
#print('trigger...')
#r.trigger()
#time.sleep(0.5)
#print('trigger...')
#r.trigger()
#time.sleep(0.5)
#print('trigger...')
#r.trigger()
#time.sleep(0.5)
#print('finishing...')
#r.stop()
        
#%%
class TDT_RZ2_Cont_Recorder(Virtual_Cont_Recorder):
    # bare bones continuous recorder. Simply uploads data from TDT and saves them in memory as
    # sweeps

    def __init__(self, circuitFile='MC-continuous-1x32atlasOmniToZIF.rcx', ephysFileName=None):
        from TDT_rpcox import RPcoX
        from zBus import ZBUSx
        super().__init__(ephysFileName=None)
        #path='C:\\Users\\colliculus\\ratCageProgramsV2\\'
        #self.circuitFile='RZ2-1x32atlasOmniToZIF_WB.rcx'
        self.circuitFile=circuitFile
        #self.circuitFile="MCtest.rcx"
        #self.TDTproject = DSPProject()
        self.zBus = ZBUSx()
        if self.zBus.ConnectZBUS('GB') == 0:
            raise ImportError('TDT ActiveX failed to connect to ZBUS')
        self.RP = RPcoX()
        if self.RP.ConnectRZ2('GB',1) == 0:
            raise ImportError('TDT ActiveX failed to connect to RZ2')
        if self.RP.LoadCOF(self.circuitFile) == 0:
            raise ImportError('TDT ActiveX failed to load circuit {}'.format(self.circuitFile))
        self.RP.Run()
        time.sleep(0.1)
        self.sampleRate=self.RP.GetSFreq()
        # work out how many channels there are
        self.nChan=int(self.RP.GetTagVal('nChan'))
        self.bufSize=int(self.RP.GetTagVal('bufSize'))
        if self.RP.GetTagType('TriggerIdx') != 73:
            raise Exception('circuit file {} does not have suitable parameter tag "TriggerIdx". Old type?'.format(circuitFile))
        self.uploadBufferTimer=None
        self.processBufferTimer=None

    def getLatestSweepStartAndEnd(self):                   
        # sweepEnd=int(self.RP.GetTagVal('sweepEnd')) # RZ2 keeps track of when the last sweep ended (as marked by zBus trigger)
        # sweepStart=int(self.RP.GetTagVal('sweepStart'))
        # newSweepCompleted = sweepEnd > self.sweepEnd+1
        # return newSweepCompleted, sweepStart, sweepEnd
        self.latestTriggerIdx=int(self.RP.GetTagVal('TriggerIdx'))
        if (self.latestTriggerIdx == 0):
            # no triggers yet. No sweeps to finalize.
            return False,0,0
        if self.latestTriggerIdx > self.numTriggersRead:
            # if there are new trigger times to upload, do so
            self.triggerSamples=self.RP.ReadTagVEX('TriggerSamples',0,self.latestTriggerIdx,'I32','I32',1)[0]
            self.numTriggersRead=self.latestTriggerIdx

        if self.numTriggersRead > self.numTriggersProcessed:
            if self.numTriggersProcessed<1:
                fromIdx=0
            else:
                fromIdx=self.triggerSamples[self.numTriggersProcessed-1]    
            toIdx=self.triggerSamples[self.numTriggersProcessed]
            # make sure the trigger index isn't beyond the number of samples uploaded
            if toIdx >= self.totalUploaded/self.nChan:
                return False, 0, 0
            # if you get this far it's time to process a new sweep
            self.numTriggersProcessed+=1
            # # print()
            # print('   ====> Last recording trigger index is ',self.latestTriggerIdx)
            # print('   ====> Triggers processed is ',self.numTriggersProcessed)
            # print('   Trigger samples: ',self.triggerSamples)
            print('   ====> processing sweep Nr {} from sample {} to {}'.format(self.numTriggersProcessed,fromIdx,toIdx))
            return True, fromIdx, toIdx
        else:
            return False, 0, 0
    
    def uploadData(self):
        if config.status=="abort":
            return
        # uploads data from buffer. Returns true if a new sweep was completed, false otherwise
        try:
            # add new data from RZ2 buffer to local sigBuf
            currIdx=int(self.RP.GetTagVal('ADidx'))
            # self.lastError='upload idx {}'.format(currIdx)
            bufEnd=currIdx
            if bufEnd < self.lastIdx:
                bufEnd+=self.bufSize
            nextChunkSize=bufEnd-self.lastIdx
            if nextChunkSize < 500:
                # too small, don't bother
                return
            # reads the recorder's multichannel signal
            sig=np.array(self.RP.ReadTagVEX('ADmsig',self.lastIdx,nextChunkSize,'F32','F32',1)[0]).astype('float32')
            if len(sig)==0:
                raise Exception("RP.ReadTagVEX('ADmsig',{},{},...) returned no data.".format(self.lastIdx,nextChunkSize))
            self.lastIdx=currIdx
            self.sigBuf=np.append(self.sigBuf,sig,axis=0)
            # print('sig buf dtypes :', sig.dtype, self.sigBuf.dtype )
            self.totalUploaded+=nextChunkSize
            # print('    uploaded {} samples so far'.format(self.totalUploaded)); sys.stdout.flush()
            # check if a sweep has ended.
            newSweepCompleted, sweepStart, sweepEnd = self.getLatestSweepStartAndEnd()
            while newSweepCompleted: # this tells us that the new sweepEnd is beyond the last remembered sweepEnd, so a new sweep is finished                
                # if sweepStart == sweepEnd:
                #     return # sweep of zero length. Nothing to process. 
                nextTimeStamp=sweepStart/self.sampleRate
                self.timestampBuffer.append(nextTimeStamp)
                # print('time stamp buffer now:', self.timestampBuffer)
                # print('Allocating new sweep from samples {} to {}'.format(sweepStart,sweepEnd))
                self.processNewSweep(sweepStart,sweepEnd)
                newSweepCompleted, sweepStart, sweepEnd = self.getLatestSweepStartAndEnd()
            # newSweep= self.newSweepAcquired()
            # if newSweep[0]:
            #     self.processNewSweep(int(self.RP.GetTagVal('sweepStart')),newSweep[1])
            #     return True
            # else:
            #     return False
            
        except BaseException as e:
            # make sure system timer is halted if error occurs
            config.status="abort"
            self.lastError=str(e)
            self.stop()
            print('Exception occurred. Recorder stopped.')
            raise e     
            
    def trigger(self):
        print('TDT_RZ2_Cont_Recorder.trigger() triggering on zBus.')
        self.zBus.zBusTrigA(0,0,6)
        
    def start(self):
        self.running=True
        self.RP.SoftTrg(1)
        #super().start()
        if isinstance(self.ephysFileName, str):
            # ephysFile is a file name. 
            self.ephysFileHandle = open(self.ephysFileName,'ab')
        else:
            self.ephysFileHandle = None
        self.lastIdx=0
        self.lastUploadAt=time.time()
        self.sigBuf=np.array([]).astype('float32')
        self.sweepStart=0
        self.sweepEnd=0   
        self.totalUploaded=0
        self.numTriggersProcessed=0
        self.numTriggersRead=0
        self.uploadBufferTimer=RepeatedTimer(min(0.3,self.sweepLen/2),self.uploadData)
        self.timestampBuffer=[]
        self.recordingStartedAt=time.time()
        self.uploadBufferTimer.start()
        print('recorder started.')
        sys.stdout.flush()
         
    def stop(self): 
        self.running=False
        if not self.uploadBufferTimer is None:
            self.uploadBufferTimer.stop()
        if not self.processBufferTimer is None:
            self.processBufferTimer.stop()
        time.sleep(0.2)
        self.RP.SoftTrg(4) # we send a soft trig 4 to signal a final new sweep
        self.uploadData()
        self.saveAcquiredSweeps()
        if not self.ephysFileHandle is None:
            self.ephysFileHandle.close()
            self.ephysFileHandle=None
        self.RP.SoftTrg(1)
        print('recorder stopped.')
        sys.stdout.flush()

class TDT_RZ2_TimerSave_Cont_Recorder(TDT_RZ2_Cont_Recorder):
    # runs a timer to save data to disk or process it periodically. 
    # Experience shows that this can easily cause threading / memory access issues though.
    
    def start(self):
        super().start()
        self.processBufferTimer=RepeatedTimer(3,self.saveAcquiredSweeps)
        self.processBufferTimer.start()
         

        
#%%
class TDT_fixedSweep_RZ2_Cont_Recorder(TDT_RZ2_TimerSave_Cont_Recorder):
    
    def __init__(self, ephysFileName=None, circuitFile='MC-fixedInterval-8x8-ViventiECOG_6k.rcx'):
        super().__init__(ephysFileName=ephysFileName, circuitFile=circuitFile)
        self.bufSize=int(self.RP.GetTagVal('bufSize'))
        self.maxUploadInterval=self.bufSize/self.nChan/self.sampleRate
        self.sweepsToRecord=0
        self.displayFilter=None
        
    def uploadData(self):
        # uploads data from buffer.
        if self.sweepsRecorded>=self.sweepsToRecord:  
            return False # we already uploaded all teh data we need.
        try:
            #% add new data from RZ2 buffer to local sigBuf
            self.currIdx=int(self.RP.GetTagVal('ADidx'))
            # bear in mind that currIdx may wrap around when buffer is full.
            # We therefore compute a bufEnd which keeps track of a possible wrap around
            # Multiple wrap arounds are not considered, but doing so would not be useful.
            bufEnd=self.currIdx
            if bufEnd < self.lastIdx:
                bufEnd+=self.bufSize
            nextChunkSize=bufEnd-self.lastIdx
            # if nextChunkSize < 500:
            #     # too small, don't bother
            #     return
            
            # keep an eye on time elapsed between uploads
            now=time.time()
            timeSinceLastUpload=now-self.lastUploadAt
            self.lastUploadAt=now
            print('Last upload {:3.2f} seconds ago'.format(timeSinceLastUpload))
            if timeSinceLastUpload > self.maxUploadInterval:
                print()
                print('***')
                print('ERROR: data upload rate not keeping up with buffer!')
                print('***')
                print()
                self.stop()
                raise Exception('Buffer overrun')
            # read the recorder's multichannel signal
            sig=np.array(self.RP.ReadTagVEX('ADmsig',self.lastIdx,nextChunkSize,'F32','F32',1)[0]).astype('float32')
            if len(sig)==0:
                raise Exception("RP.ReadTagVEX('ADmsig',{},{},...) returned no data.".format(self.lastIdx,nextChunkSize))
            self.sigBuf=np.append(self.sigBuf,sig,axis=0)
            self.totalUploaded+=nextChunkSize
            #%
            self.lastIdx=self.currIdx
            # check if a sweep has ended.
           
            if self.sweepsRecorded<self.sweepsToRecord: 
                while  self.totalUploaded >= self.nextSweepEnd():
                    self.timestampBuffer.append(self.nextSweepEnd()/self.nChan/self.sampleRate)
                    self.processNewSweep(0, self.currentSweepISIsamples() )
                    self.nextSweepStart=self.nextSweepEnd()
                    self.sweepsRecorded+=1
                    if self.sweepsRecorded>=self.sweepsToRecord:        
                        self.stop()
                        break
                # check if we acquired enough sweeps. If so, halt recording
                print('Recorded sweep {} out of {}. (ADidx is {})'.format(self.sweepsRecorded,self.sweepsToRecord, self.currIdx))

        except BaseException as e:
            # make sure system timer is halted if error occurs
            self.lastError=str(e)
            self.stop()
            raise e          
            
    def nextSweepEnd(self):
        return self.nextSweepStart+ self.currentSweepISIsamples()*self.nChan
    
    # def trigger(self):
    #     self.zBus.zBusTrigA(0,0,6)

    def currentSweepISIsamples(self):
        if type(self.ISIsamples) ==int:
            return self.ISIsamples
        else:
            return self.ISIsamples[self.sweepsRecorded]
        
    def start(self, ISIsamples, sweepsToRecord):
        # ISIsamples can be a list 
        self.ISIsamples=ISIsamples
        self.sweepsRecorded=0
        self.sweepLen=self.currentSweepISIsamples()/self.sampleRate
        self.sweepsToRecord=sweepsToRecord
        self.nextSweepStart=0
        self.lastIdx=0
        self.sigBuf=np.array([]).astype('float32')
        self.totalUploaded=0
        if isinstance(self.ephysFileName, str):
            # ephysFile is a file name. 
            self.ephysFileHandle = open(self.ephysFileName,'ab')
        else:
            self.ephysFileHandle = None
        print('Starting fixed sweep size recorder to record {} sweeps of {} samples ({} s) each.'.format(\
                     self.sweepsToRecord,self.ISIsamples,self.sweepLen))
        # self.sweepEndSamples=(np.linspace(1,self.sweepsToRecord,self.sweepsToRecord)*self.ISIsamples).astype(int)
        self.zBus.zBusTrigA(0,0,6)
        self.uploadBufferTimer=RepeatedTimer(0.5,self.uploadData)
        self.processBufferTimer=RepeatedTimer(2,self.saveAcquiredSweeps)
        self.recordingStartedAt=time.time()
        self.lastUploadAt=self.recordingStartedAt
        self.uploadBufferTimer.start()
        self.processBufferTimer.start()
        self.timestampBuffer=[0]
        print('recorder started.')
        sys.stdout.flush()
        




#%%   # the sweep recorded by TDT
class MCsweep:    
    def __init__(self, source=None, sweeplen=1000, srate=0, tStamp=0.0): 
        # initialize empty sweep
        self.signal=None
        self.timeStamp=tStamp
        self.sampleRate=srate
        if isinstance(source,np.ndarray):
            self.signal=source.reshape(sweeplen,int(len(source)/sweeplen))           
        if isinstance(source,tuple):
            self.signal=np.array(source).astype('float32').reshape(sweeplen,int(len(source)/sweeplen))           
        if isinstance(source, (int, float)):
            self.sampleRate=source            
        if hasattr(source, 'getSweep'):             
            # read in sweep from TDT device
            self.signal= source.getSweep()
            self.sampleRate=source.sampleRate
        if isinstance(source, IOBase):
            # read sweep from file
            sRate, tStamp, sig = readMchanSignalFromFile(source)
            self.signal=sig
            self.timeStamp=tStamp
            self.sampleRate=sRate
            
    def makeSineSweep(self,freqs=[100,250,500,1000], t_offset=0, sweeplen=0.100,srate=10000):
        # constructor for sine sweep objects used for testing filtering 
        period=1/srate
        Nsamples=int(sweeplen*srate)
        Nchan=len(freqs)
        self.sampleRate=srate
        self.signal=np.zeros((Nsamples,Nchan)).astype('float32')
        t=np.linspace(0,Nsamples-1,num=Nsamples)*period+t_offset
        for cc in range(Nchan):
            self.signal[:,cc]=np.cos(2*np.pi*freqs[cc]*t)
            
    def duration(self):
        return self.signal.shape[0]/self.sampleRate

    def nyquist(self):
        return self.sampleRate/2
            
    def spikeFilterCoeffs(self):
        b,a = butter(4,(300/self.nyquist(), 3000/self.nyquist()),'bandpass')
        return [b,a]
    
    def LFPfilterCoeffs(self, lowpass=300, highpass=60):
        fs = 24414.0625
        bNotch, aNotch = iirnotch(50, 30, fs)
        b,a = butter(4,(highpass/self.nyquist(), lowpass/self.nyquist()),'bandpass')
        return [b,a]
#        return [[b,a], [bNotch, aNotch]]
    
    def AMUAFilterCoeffs(self, lowpass=6000):
        bBand,aBand = butter(4,(300/self.nyquist(), 6000/self.nyquist()),'bandpass')
        bLow,aLow = butter(4,(lowpass/self.nyquist()),'lowpass')
        return [[bBand,aBand], [bLow,aLow]]    

    def IIRfilterFlipped(self,coefs, trimEnd=200):
        # make a filtered copy of the sweep 
        # with IIR filter with coefficients b, a
        result=MCsweep(self.sampleRate)
        result.timeStamp=self.timeStamp
        result.signal, zo =np.flipud(lfilter(coefs[0],coefs[1],np.flipud(self.signal),axis=0))
        if trimEnd > 0:
            result.signal=result.signal[0:-trimEnd,:]
        return result, zo

    def IIRfilter(self,coefs, zi=None):
        # make a filtered copy of the sweep 
        # with IIR filter with coefficients b, a
        result=MCsweep(self.sampleRate)
        result.timeStamp=self.timeStamp
        Nchans=self.signal.shape[1]
        if zi is None:
            zi=[]
            zi_base=lfilter_zi(coefs[0],coefs[1])
            for chan in range(Nchans):
                #zi.append(zi_base)
                zi.append(np.zeros((len(coefs[1])-1,)))
        result.signal=np.zeros(self.signal.shape).astype('float32')
        zo=[]
        for chan in range(Nchans):
            y, zo_c=lfilter(coefs[0],coefs[1],self.signal[:,chan],axis=0,zi=zi[chan])
            result.signal[:,chan]=y
            zo.append(zo_c)

        return result
    
    def calcLFP(self,coefs=[], padLen=300, downsample=10):
        # make a AMUA copy of the sweep 
        # Calculates analog multiunit activity (AMUA) measure as in periodic gerbil paper.
        # See http://journal.frontiersin.org/article/10.3389/fncir.2015.00037/full
        result=MCsweep(self.sampleRate/downsample) # decimation will make AMUA have half orig sample rate
        result.timeStamp=self.timeStamp
        if coefs==[]:
            coefs=self.LFPfilterCoeffs()
#        # We pad the signal with even symmetry copies
#        head=-np.flipud(self.signal[0:padLen,:])
#        headShift=head[-1,:]-self.signal[0,:]
#        head=head-headShift
#        tail=-np.flipud(self.signal[-padLen:,:])
#        tailShift=tail[0,:]-self.signal[-1,:]
#        tail=tail-tailShift
#        insig=np.vstack([head, self.signal, tail])
#        insig=filtfilt(coefs[0],coefs[1],insig,axis=0)
#        insig=filtfilt(coefs[0],coefs[1],insig,axis=0)
#        result.signal=insig[padLen:-padLen,:]
        result.signal=filtfilt(coefs[0],coefs[1],self.signal,axis=0,padlen=padLen)
#        result.signal=filtfilt(coefs[0][0],coefs[0][1],self.signal,axis=0,padlen=padLen)
#        result.signal=filtfilt(coefs[1][0],coefs[1][1],self.signal,axis=0,padlen=padLen)
        # downsample
        result.signal=decimate(result.signal,downsample,axis=0)
        return result
    
    def sigSnip(self,t, rest=False):
        # returns section of the signal covering time window t[0] to t[1] in seconds
        # if "rest" is True, it returns the section *outside* the window
        t0=np.floor(np.round(t[0]*self.sampleRate)).astype(int)
        if t[1]== -1:
            # t[1] of -1 means "to the end
            if rest:
                return self.signal[0:t0,:]
            else:
                return self.signal[t0:,:]
        else:
            t1=int(np.floor(t[1]*self.sampleRate))
            if rest:
                s=self.signal[0:t0]
                if self.signal.shape[0]-t1 > 0:
                    s=np.vstack((s,self.signal[t1:]))
                return s
            else:
                return self.signal[range(t0,t1),:]

    def interpBlanking(self,x1,x2):
        # blanks out a presumed aretefact from sample x1 to sample x2 inclusive 
        # by replacing the stretch [x1,x2] with linear interpolation 
        numSamples=x2-x1
        if numSamples < 0:
            raise Exception("MCsweep.interpBlanking({},{}) : stretch to interpolate must be positive number.".format(x1,x2))
        for chan in range(self.signal.shape[1]):
            y1=self.signal[x1,chan]
            y2=self.signal[x2,chan]
            snip=np.linspace(y1,y2,numSamples)
            self.signal[x1:x2,chan]=snip

    def calcAMUA(self,coefs=[], padLen=300, downsample=2, FsNew=0):
        # make a AMUA copy of the sweep 
        # Calculates analog multiunit activity (AMUA) measure as in periodic gerbil paper.
        # See http://journal.frontiersin.org/article/10.3389/fncir.2015.00037/full
        result=MCsweep(self.sampleRate/downsample if FsNew==0 else FsNew) # decimation will make AMUA have half orig sample rate
        result.timeStamp=self.timeStamp
        if coefs==[]:
            coefs=self.AMUAFilterCoeffs()
        bpCoefs=coefs[0]
        lpCoefs=coefs[1]
#        # We pad the signal with even symmetry copies
#        head=-np.flipud(self.signal[0:padLen,:])
#        headShift=head[-1,:]-self.signal[0,:]
#        head=head-headShift
#        tail=-np.flipud(self.signal[-padLen:,:])
#        tailShift=tail[0,:]-self.signal[-1,:]
#        tail=tail-tailShift
#        insig=np.vstack([head, self.signal, tail])
#        insig=filtfilt(bpCoefs[0],bpCoefs[1],insig,axis=0)
#        insig=np.abs(insig)
#        insig=filtfilt(lpCoefs[0],lpCoefs[1],insig,axis=0)       
#        result.signal=insig[padLen:-padLen,:]
        insig=filtfilt(bpCoefs[0],bpCoefs[1],self.signal,axis=0,padlen=padLen)
        insig=np.abs(insig)
        result.signal=filtfilt(lpCoefs[0],lpCoefs[1],insig,axis=0, padlen=padLen)          
        # downsample
        result.signal=decimate(result.signal,downsample,axis=0)
        return result
    
    def RMS(self, rwin):
        # returns root mean square value of signal amplitude over the response window rwin
        return np.sqrt(np.mean(self.sigSnip(rwin)**2,axis=0))

    def response(self, rwin,swin=[]):
        # returns a "response measure" as the mean signal amplitude over the response window rwin
        #  optionally baseline corrected by response in spon window swin
        if swin == 'rest':
            spon = np.mean(self.sigSnip(rwin,rest=True),axis=0)
        else:            
            if swin ==[]:
                spon = 0
            else:
                spon = np.mean(self.sigSnip(swin),axis=0)
        return np.mean(self.sigSnip(rwin),axis=0)-spon

    def saveSweep(self,ephysFile):
        if isinstance(ephysFile, IOBase):
            # ephysFile is a file handle. 
            writeMchanSignalToFile(ephysFile,self.sampleRate, self.timeStamp, self.signal)
            return
        if isinstance(ephysFile, str):
            # ephysFile is a file name. 
            fileObject = open(ephysFile,'ab')
            writeMchanSignalToFile(fileObject,self.sampleRate, self.timeStamp, self.signal)
            fileObject.close()
            return
        # if you get to this point ephysFile is not somethign we can work with
        raise IOError('MCsweep.saveSweep() must be given file handle or file name string.')
        
    def plotSig(self, nCol=8, pltHandle=plt, channels=[], lineObjects=[]): # plot a multichannel signal
        if self.signal is None:
            print('Signal empty. Nothing to plot.')
            return
        # matplotlib is memory leaky. We have to reuse the line objects to prevent memory overflow errors.
        # No lineobjects: collect them
        nChan=self.signal.shape[1]
        if nChan<nCol:
            nCol=round(nChan/2)
        if len(lineObjects)==0:
            pltHandle.clf()
            ax=pltHandle.gca()
            if channels ==[]:
                channels=range(nChan)
            for chan in channels:
                line, =ax.plot([],[])
                lineObjects.append(line)
        ax=pltHandle.gca()
        # first work out how many rows of panels we need
        #nRow= np.ceil(nChan/nCols)
        # now work out how big our x and y offsets need to be
        xoffset=self.signal.shape[0]*0.1
        yrange=np.max(np.abs(self.signal[20:,:]))
        yoffset=yrange*2.2
        if yoffset==0:
            yoffset=1
        xpoints=np.arange(self.signal.shape[0])
        # now plot, starting from the bottom left
        row=0
        col=0
        if channels ==[]:
            channels=range(nChan)
        for chan in channels:
            lineObjects[chan].set_data((xpoints+(xoffset+xpoints[-1])*col)/self.sampleRate,(self.signal[:,chan]-yoffset*row)*1000)
            col+=1
            if col >= nCol:
                col=0
                row+=1
        ax.set_xlabel('time (s)'); ax.set_ylabel('mV');
        ax.set_ylim(np.array([-(yoffset*(row-1)+yrange),yrange])*1000)
        ax.set_xlim([0,(xpoints[-1]+xoffset)*nCol/self.sampleRate])
        return lineObjects
        

        
#%%
#def testSignalWriting():
#    fname='c:/temp/test.ephys'
#    fileObject = open(fname,'wb')
#    sweep=np.vstack((np.array(range(100)),np.random.random(100),np.array(range(100,0,-1))))
#    sweep=sweep.transpose()
#    writeMchanSignalToFile(fileObject,20.2,sweep)
#    writeMchanSignalToFile(fileObject,40.4,sweep)
#    writeMchanSignalToFile(fileObject,60.6,sweep)
#    fileObject.close()
#

#def sweepsToAMUA(swps,coefs=[], downsample=2,):
#    # make a AMUA copy of the sweep 
#    # Calculates analog multiunit activity (AMUA) measure as in periodic gerbil paper.
#    # See http://journal.frontiersin.org/article/10.3389/fncir.2015.00037/full
#    result=MCsweep(self.sampleRate/downsample) # decimation will make AMUA have half orig sample rate
#    result.timeStamp=self.timeStamp
#    if coefs==[]:
#        coefs=swps.AMUAFilterCoeffs()
#    # first apply bandpass filter
#    bpCoefs=coefs[0]
#    # work out required padlen. Let's work with a 30 ms pad
#    padLen=int(np.ceil(self.sampleRate*0.03))
#    #result.signal=np.flipud(lfilter(bpCoefs[0],bpCoefs[1],np.flipud(self.signal),axis=0))
#    result.signal=filtfilt(bpCoefs[0],bpCoefs[1],self.signal,axis=0, padlen=padLen, padtype="odd")
#    # take absolute value
#    result.signal=np.abs(result.signal)
#    # low pass
#    lpCoefs=coefs[1]
#    #result.signal=np.flipud(lfilter(lpCoefs[0],lpCoefs[1],np.flipud(result.signal),axis=0))
#    result.signal=filtfilt(lpCoefs[0],lpCoefs[1],result.signal,axis=0, padlen=padLen, padtype="even")
#    # trim discontinuity artifacts if desired. This should not happen with filtfilt
#    if trimEnd > 0:
#        result.signal=result.signal[0:-trimEnd,:]
#    # downsample
#    result.signal=decimate(result.signal,downsample,axis=0)
#    return result

def poolSweeps(swp):
    # "pool" (average) the responses (signals) of all the MSweeps in input list "swp"
    
    #signals in swp may differ in length. Can only pool over the shortest length in teh collection
    alen=np.zeros([len(swp)])
    for ss in range(len(swp)):
        alen[ss]=swp[ss].signal.shape[0]
        #print('sweep {}: length {}'.format(ss, swp[ss].signal.shape[0]))
        #pooledPSTH+=swp[ss].signal
    minLen=int(np.min(alen))
    
    result=MCsweep(swp[0].sampleRate) 
    result.signal=swp[0].signal[0:minLen,:].copy()
    for ss in range(1,len(swp)):
        result.signal+=swp[ss].signal[0:minLen,:]
    result.signal/=len(swp)
    return result
        
def testSignalReading():
    #fname='c:/temp/ephysTest/a1.ephys'
    fname='/home/colliculus/electrophys/TWF_IC_recording/NormalHearingNonTrain/ctrl_1805/data/ctrl_1805_TWF300Hz_2_P1.ephys'
    swps, stimPar=readEphysFile(fname)
    return swps, stimPar

def writeEphysFile(fname,swps,stim=None):
    # check file extension
    if fname[-6:]!='.ephys':
        fname=fname+'.ephys'        
    # read in .ephys file
    fileObject = open(fname,'wb')
    for swp in swps:
        swp.saveSweep(fileObject)
    fileObject.close()
    if not stim is None:
        # write corresponding stim param table
        if type(stim)==pd.core.frame.DataFrame:
            fname=fname[:-6]+'.csv'
            stim.to_csv(fname,',')

def readEphysFile(fname):
    # check file extension
    if fname[-4:]=='.csv':
        fname=fname[:-4]
    if fname[-6:]!='.ephys':
        fname=fname+'.ephys'        
    swps=np.array([])
    # read in .ephys file
    with open(fname,'rb') as fileObject:
        swp=MCsweep(fileObject)
        while not swp.signal is None:
            swps=np.append(swps,swp)
            swp=MCsweep(fileObject)
    #read in corresponding stim param table
    fname=fname[:-6]+'.csv'
    stimPar=readEphysStimTable(fname)
    return swps, stimPar

def readEphysStimTable(fname):
    if fname[-6:]=='.ephys':
        fname=fname[:-6]+'.csv'        
    if fname[-4:]!='.csv':
        fname=fname+'.csv'        
    try:
        stimPar=pd.read_csv(fname,',')
        stimPar=stimPar.loc[:, ~stimPar.columns.str.contains('^Unnamed')]
    except:
        print('warning, no stim file found for '+fname)
        stimPar=None
    return stimPar
        
def readMchanSignalFromFile(fileObject):
    try:
        tStamp=struct.unpack("f",fileObject.read(4))[0]
        sampleRate=struct.unpack("f",fileObject.read(4))[0]
        sweeplen=struct.unpack("i",fileObject.read(4))[0]
        numChan=struct.unpack("i",fileObject.read(4))[0]
        sweep=None
        for chanIdx in range(numChan):
            achan=np.array(struct.unpack("{}f".format(sweeplen),fileObject.read(4*sweeplen))).astype('float32')
            if sweep is None:
                sweep=np.array(achan)
            else:
                sweep=np.vstack((sweep,np.array(achan)))
        return sampleRate, tStamp, sweep.transpose()
    except BaseException as e:
        #print('Finished reading sweeps with message: ',str(e))
        # reading file failed, probably EOF
        return None, None, None

def writeMchanSignalToFile(fileObject,sampleRate, timeStamp, sweep):
    fileObject.write(bytearray(struct.pack("f",timeStamp)))
    fileObject.write(bytearray(struct.pack("f",sampleRate)))
    sweeplen=sweep.shape[0]
    fileObject.write(bytearray(struct.pack("i",sweeplen)))
    fileObject.write(bytearray(struct.pack("i",sweep.shape[1])))
    for chanIdx in range(sweep.shape[1]):
        achan=sweep[:,chanIdx].astype('float32').tolist()
        fileObject.write(bytearray(struct.pack("{}f".format(sweeplen),*achan)))
    fileObject.flush()
        
#%%
def multiPlot(x,y,ncols, marker='', linestyle='-'):
    axHnds=tuple()
    pltHnds=tuple()
    plt.clf()
    # first work out how many rows of panels we need
    nchan=y.shape[1]
    nrows=np.ceil(nchan/ncols)
    yrange=[np.min(y), np.max(y)]
    for ii in range(nchan):
        # collect the axes and line handles and return them
        axHnds+= (plt.subplot(nrows,ncols,ii+1),)
        pltHnds+= (plt.plot(x, y[:,ii], marker=marker, linestyle=linestyle),)
        if not np.mod(ii,ncols) == 0:
            plt.yticks([])
        if not ii>=nchan-ncols:
            plt.xticks([]),
        plt.ylim(yrange)
    return axHnds, pltHnds