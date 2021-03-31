'''
Modulation Recognition with Deep Artificial Neural Networks
Analog Modulation
Warning: Those that attempt to run this code using a CPU should
expect a signficant wait time for the model to learn.
'''
''' TO DO: Use RML2016.04B as the validation set for GUI'''
''' TO DO: Apply L1 and L2 regulizers and sparsity'''
# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,sys,random
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["KERAS_BACKEND"] = "theano"
import numpy as np
import tensorflow as T
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as cPickle
import keras

# Load the dataset
#  You will need to seperately download or generate this file
Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'),encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
analog_mods = ['WBFM', 'AM-SSB','AM-DSB']
for mod in analog_mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Split Data into Analog and Digital Signals
#analog = []
#analog_mods = ['WBFM', 'AM-SSB','AM-DSB']

#for mod in analog_mods:
#    for snr in snrs:
#        analog.append(Xd[(mod,snr)])
#        for i in range(Xd[(mod,snr)].shape[0]): lbl.append((mod,snr))
#analog = np.vstack(analog)

# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
#try editing the nTrain
np.random.seed(2018)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy = list(yy)
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(map(lambda x: analog_mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: analog_mods.index(lbl[x][0]), test_idx))

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = analog_mods

# Initialize ANN
analog_classifier = Sequential()
analog_classifier.add(Reshape(in_shp + [1], input_shape=in_shp))
analog_classifier.add(Flatten())
# Stack Dense Layers with Dropout #try adding a dropout function
analog_classifier.add(Dense(500,kernel_initializer='glorot_uniform',
    activation="relu"))
analog_classifier.add(Dense(1500, kernel_initializer='glorot_uniform', activation="relu"))
#add a dropout function after this


# Output Layer
analog_classifier.add(Dense( len(classes), kernel_initializer='he_normal', name="dense4" ))
analog_classifier.add(Activation('softmax'))
analog_classifier.add(Reshape([len(classes)]))

# Compliing the CNN # can change 'adam' to either adadelta, adagrad, rmsprop
analog_classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
analog_classifier.summary()

# Set up some params #generally more epochs is better learning, avoid overfitting, mostly focus on ann not cnn
nb_epoch = 100 # number of epochs to train on
batch_size = 1024  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'ann_analog.h5'
''' EarlyStopping'''
history = analog_classifier.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])

# we re-load the best weights once training is finished
analog_classifier.load_weights(filepath)

# Evaluating the CNN
# Show simple version of performance
score = analog_classifier.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(score)

# Show loss curves
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()