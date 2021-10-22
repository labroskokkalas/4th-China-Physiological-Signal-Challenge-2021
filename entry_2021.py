#!/usr/bin/env python3

import numpy as np
import os
import sys
import glob
import json
import math
import wfdb
from scipy import signal
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""
Written by:  Lampros Kokkalas, Nikolas A. Tatlas, Stelios M. Potirakis
             Department of Electrical and Electronics Engineering, 
             University of West Attica, Athens, Greece
             lkokkalas@uniwa.gr
"""

# define convolutional recurrent network model
def stateful_model(input_shape):
     X_input = tf.keras.Input(batch_shape = input_shape)
     X = tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=2)(X_input)                                
     X = tf.keras.layers.Activation("relu")(X)                                 
     X = tf.keras.layers.Dropout(rate=0.4,noise_shape=(1, 1, 32))(X)
     X = tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=2)(X)                                
     X = tf.keras.layers.Activation("relu")(X)                                 
     X = tf.keras.layers.Dropout(rate=0.4,noise_shape=(1, 1, 32))(X)
     X = tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=2)(X)                                 
     X = tf.keras.layers.Activation("relu")(X)                                 
     X = tf.keras.layers.Dropout(rate=0.4,noise_shape=(1, 1, 32))(X)     
     X = tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=2)(X)                                 
     X = tf.keras.layers.Activation("relu")(X)                                
     X = tf.keras.layers.Dropout(rate=0.4,noise_shape=(1, 1, 32))(X)        
     X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32, return_sequences=True, stateful=True, reset_after=True),merge_mode='ave')(X)
     X = tf.keras.layers.Dropout(rate=0.4,noise_shape=(1, 1, 32))(X)                                  
     X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32, return_sequences=True, stateful=True, reset_after=True),merge_mode='ave')(X)
     X = tf.keras.layers.Dropout(rate=0.4,noise_shape=(1, 1, 32))(X)  
     X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32, return_sequences=True, stateful=True, reset_after=True),merge_mode='ave')(X)
     X = tf.keras.layers.Dropout(rate=0.4,noise_shape=(1, 1, 32))(X)                                  # dropout (use 0.8)
     X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32, return_sequences=True, stateful=True, reset_after=True),merge_mode='ave')(X)
     X = tf.keras.layers.Dropout(rate=0.4,noise_shape=(1, 1, 32))(X)      
     X = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = "sigmoid"))(X) 
     model = tf.keras.Model(inputs = X_input, outputs = X)
     return model   

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False)

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']
    return sig, length, fs        

def export_data(sample_path, result_path, frame_length, Tx):
    seperator = os.sep
    sig, sig_len, fs = load_data(sample_path)
    # pad data sequencies to 60 sec frames
    if sig_len/fs < frame_length:
        sig_pad = sig[:]
        for i in range(math.ceil(frame_length/(sig_len/fs))-1):
            sig_pad = np.concatenate([sig_pad, sig_pad])
        sig_pad = sig_pad[0:frame_length*fs]   
    else:        
        pad_remainder = fs*frame_length-int(sig_len%(fs*frame_length))
        sig_pad = np.concatenate([sig, sig[0:pad_remainder]])
 
    # export time domain data
    dataBufs = {}
    dataBufs["signal1"] = sig_pad[:,0]
    #dataBufs["signal1_spectrogram"] = sig_pad[:,0]
    dataBufs["signal2"] = sig_pad[:,1]
    #dataBufs["signal2_spectrogram"] = sig_pad[:,1]
    for key in list(dataBufs.keys()):
            if not os.path.exists(result_path+'/'+key):
                os.mkdir(result_path+'/'+key) 
            if not os.path.exists(result_path+'/'+key+'/'+sample_path.split(seperator)[-1]):
                os.mkdir(result_path+'/'+key+'/'+sample_path.split(seperator)[-1])
            maxData = max(dataBufs[key])
            nperseg = 1
            n_freq = 1
            if key.endswith('_spectrogram'):
                nperseg = 10
                n_freq = 6                        
            X = np.zeros([int(len(dataBufs[key])/(fs*frame_length)), Tx, n_freq]) 
            for i in range(int(len(dataBufs[key])/(fs*frame_length))):    
                dataSeg = dataBufs[key][i*(fs*frame_length):(i+1)*(fs*frame_length)]
                dataSeg = np.multiply(dataSeg, 1/maxData)
                if n_freq == 1 :
                    SxxFinal  = np.zeros([1,fs*frame_length])
                    SxxFinal[0] = dataSeg
                    SxxFinal = signal.resample(SxxFinal, Tx, axis=1)
                    SxxFinal = SxxFinal.T      
                    X[i] = SxxFinal.astype('float32')                    
                else:
                    # generate spectrogram
                    f, t, Sxx = signal.spectrogram(dataSeg, fs, window=('hamming'),nperseg=nperseg )      
                    SxxFinal = signal.resample(Sxx, Tx, axis=1)
                    SxxFinal = SxxFinal.T      
                    SxxFinal[SxxFinal == 0] = 1e-20
                    SxxFinal = np.log(abs(SxxFinal)) 
                    X[i] = SxxFinal.astype('float32')
            # normalize data
            normData = X[0]
            for i in range(1,X.shape[0]):
                normData = np.concatenate((normData, X[i]))
            mean = np.zeros(normData.shape[1])
            std = np.zeros(normData.shape[1])
            for i in range(normData.shape[1]):
                mean[i] = np.mean(normData[:,i])  
                std[i] = np.std(normData[:,i]) 
                if std[i] == 0.0:
                    std[i] = 1 
            dataBufs[key] = (X-mean)/std  
            # save data per frame
            for i in range(dataBufs[key].shape[0]):  
                data = dataBufs[key][i].astype('float32')  
                np.save(result_path+'/'+key+'/'+sample_path.split(seperator)[-1]+'/'+sample_path.split(seperator)[-1]+'_'+str(i*frame_length)+'_'+str((i+1)*frame_length)+".npy", data)                   
    
def remove_consecutive_ones(y, lowerLimit) :  
    for i in range(y.shape[0]):
        if (y[i] == 0 or (y[i] == 1 and i == 0)) and i < y.shape[0]-1 and y[i+1] == 1 :
            for j in range(i+2,y.shape[0]) :
                if y[j] == 0 or j == y.shape[0]-1:
                    if j-i-1 <= lowerLimit and j < y.shape[0]-1 :
                        for k in range(i+1,j+1):
                            y[k] = 0
                        if i == 0:
                            y[i] = 0                         
                    break
    for i in range(y.shape[0]):
        if y[i] == 1 and i < y.shape[0]-1 and y[i+1] == 0 :
            for j in range(i+2,y.shape[0]) :
                if y[j] == 1 or j == y.shape[0]-1:
                    if j-i-1 <= lowerLimit and j < y.shape[0]-1 :
                        for k in range(i+1,j+1):
                            y[k] = 1                        
                    break
    return y 

def run_data(model, sample_path, result_path, bio_signals, Tx, Ty, n_channels):
      seperator = os.sep
      model.reset_states()
      test_samples = glob.glob(result_path+'/'+bio_signals[0]+'/'+sample_path.split(seperator)[-1]+"/*.npy")
      test_samples.sort(key=lambda x: int(float(x.split(seperator)[-1].split("_")[3])))
      y_pred = np.zeros([len(test_samples), Ty, 1])
      for test_sample in test_samples:
          X_pred = np.zeros([1, Tx, n_channels])
          for signal in bio_signals:
              signal_data = np.load(result_path+'/'+signal+'/'+sample_path.split(seperator)[-1]+'/'+test_sample.split(seperator)[-1])
              if signal == bio_signals[0]:
                  data = signal_data
              else:    
                  data = np.concatenate((data,signal_data),axis=1)
          X_pred[0,] = data
          # initialize model
          if test_sample == test_samples[0]:
              ss = model.predict(X_pred,verbose=0)[0] 
              ss = model.predict(X_pred,verbose=0)[0] 
          # run inference 
          y_pred[test_samples.index(test_sample)] = model.predict(X_pred,verbose=0)[0]          
      
      y_pred = 1*(y_pred > 0.4)
      y_pred = y_pred.flatten()
      # remove very short detected events
      y_pred = remove_consecutive_ones(y_pred,1.0*Ty/frame_length)
      
      sig, signal_len, fs = load_data(sample_path)
      signal_end_point = int(((int(signal_len)/fs)/frame_length)*Ty)
      end_points = []
      if len([el for el in y_pred if el == 1]) > 15 and len([el for el in y_pred if el == 1]) > signal_end_point/100:
          if abs(len(y_pred) - len([el for el in y_pred if el == 1])) < 15 or (abs(len([el for el in y_pred if el == 1])-signal_end_point) < signal_end_point/100):
              end_points.append([0, int(signal_len)-1])
          else:      
              for i in range(signal_end_point):
                  if (y_pred[i] == 0 or i == 0) and i < signal_end_point-1 and y_pred[i+1] == 1 :
                      start_point = int(((i+1)/Ty)*frame_length*fs)
                  if y_pred[i] == 1 and i > 0 and ((i < signal_end_point-1 and y_pred[i+1] == 0) or i == signal_end_point-1) :
                      end_point = min(int(((i+1)/Ty)*frame_length*fs), int(signal_len)-1)
                      end_points.append([start_point, end_point])                      
      pred_dict = {'predict_endpoints': end_points}
      save_dict(os.path.join(result_path, sample_path.split(seperator)[-1]+'.json'), pred_dict)

if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    frame_length = 60
    Tx = 12000
    Ty = 747
    n_channels = 2
    bio_signals = ['signal1', 'signal2']
    
    model = stateful_model(input_shape = (1, Tx, n_channels))    
    model.load_weights('./model_v7.h5')
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)    
        sample_path = os.path.join(DATA_PATH, sample)
        export_data(sample_path, RESULT_PATH, frame_length, Tx)
        run_data(model, sample_path, RESULT_PATH, bio_signals, Tx, Ty, n_channels)

