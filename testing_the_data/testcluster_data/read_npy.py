from glob import glob
import numpy as np

import os
data_files = glob(os.path.join(os.getcwd(), 'data', 'data-*.npy'))
label_files = glob(os.path.join(os.getcwd(), 'data', 'label-*.npy'))


print(len(data_files))


for i in range(len(data_files)):
    data_file = 'data-'+str(i+1) + '.npy'
    label_file = 'label-'+str(i+1) + '.npy'
    data = os.path.join(os.getcwd(), 'data', data_file)
    label = os.path.join(os.getcwd(), 'data', label_file)
    
    data_= np.load(data)
    label_= np.load(label)
    print('data_ : ', data_.shape)
    print('label_ : ', label_.shape)
    #print('label_', label_)
'''
for data_file in data_files:
    data_= np.load(data_file)
    #print('data_ type : ', type(data_), data_.shape)
    #print(data_)
    
    
for label_file in label_files:
    label_= np.load(label_file)
    #print('Label type : ', type(label_), label_.shape)
    #print(label_)
    
'''