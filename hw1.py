#!/usr/bin/python
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras import backend as K
import pandas as pd

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

trainfile = pd.read_csv('/Users/weizhao/Downloads/train.csv')
testfile = pd.read_csv('/Users/weizhao/Downloads/test.csv')

trainfile_X = trainfile.iloc[:,1:]
testfile_X = testfile.iloc[:,:]
trainfile_Y = trainfile.iloc[:,:1]

X_train = np.asarray(trainfile_X)
X_test = np.asarray(testfile_X)

#testfile_Y = testfile.iloc[:,:]
y_train = np.asarray(trainfile_Y)
#y_test = np.asarray(testfile_Y)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Data pre-processing
# convert class vectors to binary class matrices
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0],img_rows, img_cols,1)
input_shape = (img_rows, img_cols,1)     # channels, height & width 
Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],border_mode='valid',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# Fully connected layer 1 input shape (64 * 7 * 7) = ( 3136), output shape (1024)
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

optimizer = Adadelta(lr = 0.00086)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# train the model
print('Training ------------- ')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)

score = model.predict_classes(X_test, batch_size=batch_size, verbose=0)

target = open('/Users/weizhao/Downloads/output.csv','w')

target.write('imageid,label\n')
for i in range(10000):
    target.write('%s,%s\n' % (i+1,score[i]))
    
target.close()
