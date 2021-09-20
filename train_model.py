#!/usr/bin/env python3

import numpy as np
import os
import sys
import glob
import math

import wfdb
from scipy import signal
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True 
session = InteractiveSession(config=config)

"""
Written by:  Lampros Kokkalas, Nikolas A. Tatlas, Stelios M. Potirakis
             Department of Electrical and Electronics Engineering, 
             University of West Attica, Athens, Greece
             lkokkalas@uniwa.gr
"""

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    ann_ref = wfdb.rdann(sample_path, 'atr')
    return sig, fields, ann_ref

def export_data(sample_path, run_path, frame_length, Tx):
    seperator = os.sep
    sig, fields, ann_ref = load_data(sample_path)
    end_points = []
    # pad data sequencies and annotations to 60 sec frames
    sig_len = fields['sig_len']
    fs = fields['fs']
    type = fields['comments'][0]
    sample = ann_ref.sample
    aux_note = ann_ref.aux_note
    if sig_len/200 < frame_length:
        return
    pad_remainder = fs*frame_length-int(sig_len%(fs*frame_length))
    sig_pad = np.concatenate([sig, sig[0:pad_remainder]])
    
    if pad_remainder > ann_ref.sample[0] and type == 'paroxysmal atrial fibrillation':
        ann_ref_pad_index = np.where(ann_ref.sample==[el for el in ann_ref.sample[:-1] if el < pad_remainder and ann_ref.sample[np.where(ann_ref.sample==el)[0]+1] >= pad_remainder][0])[0][0]
        sample = np.concatenate([sample, sig_len+sample[0:ann_ref_pad_index+1]]) 
        aux_note = np.concatenate([aux_note, aux_note[0:ann_ref_pad_index+1]]) 
 
    dataBufs = {}
    dataBufs["signal1"] = sig_pad[:,0]
    dataBufs["signal1_spectrogram"] = sig_pad[:,0]
    dataBufs["signal2"] = sig_pad[:,1]
    dataBufs["signal2_spectrogram"] = sig_pad[:,1]
    for key in list(dataBufs.keys()):
            if not os.path.exists(run_path+'/'+key):
                os.mkdir(run_path+'/'+key) 
            if not os.path.exists(run_path+'/'+key+'/'+sample_path.split(seperator)[-1]):
                os.mkdir(run_path+'/'+key+'/'+sample_path.split(seperator)[-1])
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
                    f, t, Sxx = signal.spectrogram(dataSeg, fs, window=('hamming'),nperseg=nperseg )      
                    SxxFinal = signal.resample(Sxx, Tx, axis=1)
                    SxxFinal = SxxFinal.T      
                    SxxFinal[SxxFinal == 0] = 1e-20
                    SxxFinal = np.log(abs(SxxFinal)) 
                    X[i] = SxxFinal.astype('float32')
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
            for i in range(dataBufs[key].shape[0]):  
                data = dataBufs[key][i].astype('float32')  
                if type == 'non atrial fibrillation':
                    np.save(run_path+'/'+key+'/'+sample_path.split(seperator)[-1]+'/'+sample_path.split(seperator)[-1]+'_'+str(i*frame_length)+'_'+str((i+1)*frame_length)+".npy", data)  
                elif type == 'persistent atrial fibrillation':
                    np.save(run_path+'/'+key+'/'+sample_path.split(seperator)[-1]+'/'+sample_path.split(seperator)[-1]+'_'+str(i*frame_length)+'_'+str((i+1)*frame_length)+"_1.npy", data)  
                else:
                    annotation = ''
                    for j in range(len(aux_note)):
                        if (aux_note[j] == '(AFL' or aux_note[j] == '(AFIB') and sample[j] >= i*fs*frame_length and sample[j] < (i+1)*fs*frame_length :
                            event_start = sample[j]/fs
                            for k in range(j+1,len(aux_note)):
                                if aux_note[k] == '(N' or k == len(aux_note)-1:
                                    if k == len(aux_note)-1:
                                        event_end = len(sig_pad)/fs
                                    else:
                                        event_end = sample[k]/fs
                                    annotation = annotation+'_'+"{:.2f}".format(event_start)+'_'+"{:.2f}".format(event_end-event_start)
                                    break
                    np.save(run_path+'/'+key+'/'+sample_path.split(seperator)[-1]+'/'+sample_path.split(seperator)[-1]+'_'+str(i*frame_length)+'_'+str((i+1)*frame_length)+annotation+".npy", data)     

def create_batch_directory(bioDir, batchDir, bio_signals, batch_size, paroxysmal):
    seperator = os.sep
    sample_dirs = [el.split(seperator)[-1 ]for el in glob.glob(bioDir+'/'+bio_signals[0]+'/*')]
    if paroxysmal :
        #train only with paroxysmal entries
        sample_dirs = ['data_25_1','data_25_10','data_25_11','data_25_12','data_25_13','data_25_14','data_25_15','data_25_16','data_25_17','data_25_18','data_25_19','data_25_2','data_25_20','data_25_21','data_25_22','data_25_23','data_25_24','data_25_3','data_25_4','data_25_5','data_25_6','data_25_7','data_25_8','data_25_9','data_32_1','data_32_10','data_32_11','data_32_12','data_32_13','data_32_14','data_32_15','data_32_16','data_32_17','data_32_18','data_32_19','data_32_2','data_32_20','data_32_21','data_32_22','data_32_23','data_32_24','data_32_25','data_32_26','data_32_27','data_32_28','data_32_29','data_32_3','data_32_4','data_32_5','data_32_6','data_32_7','data_32_9','data_39_1','data_39_10','data_39_11','data_39_12','data_39_13','data_39_14','data_39_15','data_39_16','data_39_17','data_39_18','data_39_19','data_39_2','data_39_20','data_39_21','data_39_22','data_39_3','data_39_4','data_39_5','data_39_6','data_39_7','data_39_8','data_39_9','data_48_1','data_48_10','data_48_11','data_48_12','data_48_13','data_48_14','data_48_15','data_48_16','data_48_17','data_48_18','data_48_19','data_48_2','data_48_20','data_48_21','data_48_22','data_48_23','data_48_24','data_48_3','data_48_4','data_48_5','data_48_6','data_48_7','data_48_8','data_48_9','data_101_1','data_101_10','data_101_2','data_101_3','data_101_4','data_101_5','data_101_6','data_101_7','data_101_8','data_101_9','data_104_1','data_104_10','data_104_11','data_104_12','data_104_13','data_104_14','data_104_15','data_104_16','data_104_17','data_104_18','data_104_19','data_104_2','data_104_20','data_104_21','data_104_22','data_104_23','data_104_24','data_104_25','data_104_26','data_104_27','data_104_28','data_104_3','data_104_4','data_104_5','data_104_6','data_104_7','data_104_8','data_104_9','data_60_1','data_60_10','data_60_11','data_60_12','data_60_2','data_60_3','data_60_4','data_60_5','data_60_6','data_60_7','data_60_8','data_60_9','data_64_1','data_64_10','data_64_2','data_64_3','data_64_4','data_64_5','data_64_6','data_64_7','data_64_8','data_64_9','data_66_1','data_66_10','data_66_11','data_66_12','data_66_13','data_66_14','data_66_15','data_66_16','data_66_17','data_66_18','data_66_2','data_66_3','data_66_4','data_66_5','data_66_6','data_66_7','data_66_8','data_66_9','data_68_1','data_68_10','data_68_11','data_68_12','data_68_13','data_68_14','data_68_15','data_68_16','data_68_17','data_68_18','data_68_19','data_68_2','data_68_20','data_68_21','data_68_22','data_68_23','data_68_24','data_68_25','data_68_3','data_68_4','data_68_5','data_68_6','data_68_7','data_68_8','data_68_9','data_88_1','data_88_10','data_88_11','data_88_2','data_88_3','data_88_4','data_88_6','data_88_7','data_88_8','data_88_9','data_96_1','data_96_10','data_96_11','data_96_12','data_96_13','data_96_14','data_96_15','data_96_16','data_96_17','data_96_18','data_96_19','data_96_2','data_96_20','data_96_21','data_96_22','data_96_23','data_96_3','data_96_4','data_96_5','data_96_6','data_96_7','data_96_8','data_96_9','data_98_1','data_98_10','data_98_11','data_98_12','data_98_2','data_98_3','data_98_4','data_98_5','data_98_6','data_98_7','data_98_8','data_98_9']
    sample_dirs.sort(key=lambda x: len(glob.glob(bioDir+'/'+bio_signals[0]+'/'+x+'/*')))
    reverse = False

    sample_dir_map = {}
    for dir in sample_dirs :
        sample_dir_map[dir] = [el.split(seperator)[-1 ]for el in glob.glob(bioDir+'/'+bio_signals[0]+'/'+dir+'/*.npy')]
        sample_dir_map[dir].sort(key=lambda x: int(float(x.split('_')[1])), reverse=reverse)

    batches = math.ceil(len(sample_dirs)/batch_size)
    sample_dirs.extend(sample_dirs)
    counter = 0
    for i in range(batches) :
        dirs = sample_dirs[batch_size*i:batch_size*(i+1)]
        max_file_num = max([len(sample_dir_map[el]) for el in dirs])
        for j in range(max_file_num) :
            f = open(batchDir+'/'+str(counter)+'.txt','w')
            signal = bio_signals[0]
            data = np.load(bioDir+'/'+signal+'/'+dirs[0]+'/'+sample_dir_map[dirs[0]][j%len(sample_dir_map[dirs[0]])])
            for signal in bio_signals[1:]:
                bioData = np.load(bioDir+'/'+signal+'/'+dirs[0]+'/'+sample_dir_map[dirs[0]][j%len(sample_dir_map[dirs[0]])])
                data = np.concatenate((data,bioData),axis=1)
            f.write(sample_dir_map[dirs[0]][j%len(sample_dir_map[dirs[0]])].split('.npy')[0]+'\n') 
            for dir in dirs[1:] :
                signal = bio_signals[0]
                tmp = np.load(bioDir+'/'+signal+'/'+dir+'/'+sample_dir_map[dir][j%len(sample_dir_map[dir])])
                for signal in bio_signals[1:]:
                    bioData = np.load(bioDir+'/'+signal+'/'+dir+'/'+sample_dir_map[dir][j%len(sample_dir_map[dir])])
                    tmp = np.concatenate((tmp,bioData),axis=1)
                data = np.concatenate((data,tmp)) 
                f.write(sample_dir_map[dir][j%len(sample_dir_map[dir])].split('.npy')[0]+'\n')         
            f.close()
            data32 = data.astype('float32')
            np.save(batchDir+'/'+str(counter),data32)  
            counter = counter + 1 
        
def insert_ones(y, Ty, activation_length, frame_length, segment_end_ms):
    segment_end_y = int(segment_end_ms * Ty / (1000.0*frame_length))
    # Add 1 to the correct index in the label (y)
    next_frame_counter = 0
    for i in range(segment_end_y+1, segment_end_y+activation_length+1):
        if i < Ty:
            y[i][0] = 1
        else:
            next_frame_counter = next_frame_counter + 1        
    return y, next_frame_counter 
    
class StatefulDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_files, batch_size=32, seperator = "/", Tx=1372, Ty=160, n_freq=160, frame_length=60):
        'Initialization'
        self.batch_size = batch_size
        self.Tx = Tx
        self.Ty = Ty
        self.n_freq = n_freq
        self.frame_length = frame_length
        self.seperator = seperator
        self.list_files = list_files
        self.next_frame_label = np.zeros([self.batch_size, 1])
        print("Found "+str(len(list_files))+" samples")
        self.sequences = [*range(self.batch_size)]
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.list_files)))
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index:(index+1)]
        # Find list of IDs
        list_files_temp = [self.list_files[k] for k in indexes]
        # Generate data
        X, y, z = self.__data_generation(list_files_temp)
        return X, y#, z
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_files))
        self.next_frame_label = np.zeros([self.batch_size, 1])
    def __data_generation(self, list_files_temp):
        # Initialization
        #X = np.zeros([1*self.batch_size, self.Tx, self.n_freq, 1])
        X = np.zeros([1*self.batch_size, self.Tx, self.n_freq])
        y = np.zeros([1*self.batch_size, self.Ty, 1])
        z = np.ones([1*self.batch_size, self.Ty])
        #This argument is not supported when x is a dataset, generator, or keras.utils.Sequence instance, instead provide the sample_weights as the third element of x.
        # Generate data
        ID = list_files_temp[0]
        data = np.load(ID)
        length = int(data.shape[0]/self.batch_size)
        f = open(ID.split('.npy')[0]+'.txt')
        event_files = f.readlines()
        f.close()
        for i, ID in enumerate(self.sequences):
            # Store sample
            #X[i,] = np.expand_dims(data[ID*length:(ID+1)*length], 2)
            X[i,] = data[ID*length:(ID+1)*length]
            # Store target
            start = event_files[ID].split("\n")[0].split(self.seperator)[-1].split("_")[3] 
            events = event_files[ID].split("\n")[0].split(self.seperator)[-1].split("_")[5:]
            if len(events) == 1 and events[0] == '1':
               events = [start, str(frame_length)]
            if start != "0.00" :
                    for j in range(self.Ty) :
                        if self.next_frame_label[i][0] > j :
                            y[i][j] = 1             
            self.next_frame_label[i][0] = 0
            for event in range(int(len(events)/2)):
                event_start= events[2*event]
                event_duration = events[2*event+1]
                y[i], tmp_counter = insert_ones(y[i], self.Ty, int(self.Ty*(float(event_duration))/self.frame_length), self.frame_length, 1000*(max(0,min(self.frame_length,float(event_start)-float(start)))))
                self.next_frame_label[i][0] = self.next_frame_label[i][0] + tmp_counter   
        z[y.reshape([1*self.batch_size, self.Ty]) == 1] = 20
        return X, y, z     


def stateful_model(input_shape):
     X_input = tf.keras.Input(batch_shape = input_shape)
     X = tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=2)(X_input)                                 
     X = tf.keras.layers.Activation("relu")(X)                                 
     X = tf.keras.layers.Dropout(rate=0.2,noise_shape=(1, 1, 128))(X)
     X = tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=2)(X)                                 
     X = tf.keras.layers.Activation("relu")(X)                                
     X = tf.keras.layers.Dropout(rate=0.2,noise_shape=(1, 1, 128))(X)
     X = tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=2)(X)                                 
     X = tf.keras.layers.Activation("relu")(X)                                 
     X = tf.keras.layers.Dropout(rate=0.2,noise_shape=(1, 1, 128))(X)     
     X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128, return_sequences=True, stateful=True, reset_after=True),merge_mode='ave')(X)
     X = tf.keras.layers.Dropout(rate=0.2,noise_shape=(1, 1, 128))(X)     
     X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128, return_sequences=True, stateful=True, reset_after=True),merge_mode='ave')(X)
     X = tf.keras.layers.Dropout(rate=0.2,noise_shape=(1, 1, 128))(X)       
     X = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = "sigmoid"))(X) 
     model = tf.keras.Model(inputs = X_input, outputs = X)
     return model
     
     

def train_physionet_model(samples_dir, frame_length, Tx, Ty, n_edf, batch_size):
    seperator = os.sep
    samples = glob.glob(samples_dir+"/*.npy")
    samples.sort(key=lambda x: int(x.split(seperator)[-1].split(".")[0]))
    model = stateful_model(input_shape = (1*batch_size, Tx, n_edf))    
    stateful_generator = StatefulDataGenerator(samples,batch_size=batch_size, seperator=seperator, Tx=Tx, Ty=Ty, n_freq=n_edf, frame_length=frame_length)
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    METRICS = [
      'accuracy',
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
    ]
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=METRICS)#,sample_weight_mode="temporal")
    history = model.fit(stateful_generator, 
                    epochs=200, 
                    max_queue_size=10,            
                    workers=1,                        
                    use_multiprocessing=False,       
                    shuffle=False,
                    initial_epoch=0)
    return model                

if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RUN_PATH = sys.argv[2]
    if not os.path.exists(RUN_PATH):
        os.makedirs(RUN_PATH)
    if not os.path.exists(os.path.join(RUN_PATH, 'BATCH')):
        os.makedirs(os.path.join(RUN_PATH, 'BATCH'))
        
    frame_length = 60
    Tx = 1333
    Ty = 164
    n_edf = 12
    batch_size=64
    bio_signals = ['signal1_spectrogram', 'signal2_spectrogram']
    
    training_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(training_set):
        sample_path = os.path.join(DATA_PATH, sample)
        export_data(sample_path, RUN_PATH, frame_length, Tx)    
 
    create_batch_directory(RUN_PATH, os.path.join(RUN_PATH, 'BATCH'), bio_signals, batch_size, False)
    model= train_physionet_model(os.path.join(RUN_PATH, 'BATCH'), frame_length, Tx, Ty, n_edf, batch_size)
    model.save_weights(RUN_PATH+'/model_v5.h5')
    
    create_batch_directory(RUN_PATH, os.path.join(RUN_PATH, 'BATCH_P'), bio_signals, batch_size, True)
    model= train_physionet_model(os.path.join(RUN_PATH, 'BATCH_P'), frame_length, Tx, Ty, n_edf, batch_size)
    model.save_weights(RUN_PATH+'/model_v5_p.h5')