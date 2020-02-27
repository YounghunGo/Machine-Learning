import numpy as np
#import tensorflow.compat.v1 as tf

xy = np.loadtxt('05train.txt', unpack=True, dtype='float32')

x_data = np.transpose(xy[:3])
y_data = np.transpose(xy[3:])
print (y_data.shape)
#print(y_data)