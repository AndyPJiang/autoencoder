""" trains a autoencoder neural network for noise reduction of a signal input by learning a 
representation of the signal and then reconstructing it. """

import numpy as np 
from keras.layers import Input, Dense
from keras.models import Model
import os
from matplotlib import pyplot as plt
from scipy import signal



SIG_LEN = 354
WORKDIR = '/Users/andyjiang/Desktop/data/'  

# load data 
for i in range(len(os.listdir(WORKDIR))):
    filename = os.listdir(WORKDIR)[i]
    if "x_train" in filename:
        x_train = np.load(WORKDIR + filename, allow_pickle=True)
    elif "y_train" in filename:
        y_train = np.load(WORKDIR + filename, allow_pickle=True)
    elif "x_test" in filename:
        x_test = np.load(WORKDIR + filename, allow_pickle=True)
    else:
        y_test = np.load(WORKDIR + filename, allow_pickle=True)


# normalise data
x_train = np.divide((x_train-np.amin(x_train)),(np.amax(x_train)-np.amin(x_train)))
x_test = np.divide((x_test-np.amin(x_test)),(np.amax(x_test)-np.amin(x_test)))
y_train = np.divide((y_train-np.amin(y_train)),(np.amax(y_train)-np.amin(y_train)))
y_test = np.divide((y_test-np.amin(y_test)), (np.amax(y_test)-np.amin(y_test)))


input_layer = Input(shape=(SIG_LEN,))
x = Dense(160,activation='relu')(input_layer)
x = Dense(80,activation='relu')(x)
x = Dense(40,activation='relu')(x)
encoded_x = x
x = Dense(80,activation='relu')(x)
x = Dense(160,activation='relu')(x)
x = Dense(SIG_LEN,activation='sigmoid')(x)


model = Model(input=input_layer,output=x)
model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train,y_train,batch_size=32,epochs=50,validation_data=(x_test,y_test))


test_sample = x_test[0]
test_prediction = model.predict(np.array([test_sample]))[0]


# de-normalise
test_prediction = test_prediction * (np.amax(x_test)-np.amin(x_test)) + np.amin(x_test)

plt.figure()
plt.plot(test_sample)
plt.title("Original Input")


plt.figure()
plt.plot(test_prediction)
plt.title("Autoencoder Output")


plt.show()


