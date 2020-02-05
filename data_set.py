"""
Pre-processing of data to create training and test data-sets. 
Dealing with npy files that each contain a numpy array of signal data """


from scipy import signal
import numpy as np
import os


WORKDIR = '/Users/andyjiang/Desktop/adsb_tools/sydney/' 
WORKDIR_SAVE = '/Users/andyjiang/autoencoders_data/' 

# declare constants
DATA_SIZE = len(os.listdir(WORKDIR))
NOISE_LEVEL = 1000
SIG_LEN = 354

print(DATA_SIZE)

# initialise variables
x_train = []
y_train = []
x_test = []
y_test =[]

# Currently do not upsample for computational efficiency
upsample_factor = 1


n = 0

while n < DATA_SIZE:
    filename = os.listdir(WORKDIR)[n]
    if filename.endswith(".npy"):
        try:
            a = np.load(WORKDIR + filename, allow_pickle=True)
        except:
            n+=1
            continue
        
        # returns dict_keys(['date', 'DF', 'ICAO', 'data', 'IQ', 'RSSI', 'error','andy])
        # sig is in complex format
        upsig = a.item()['IQ']


        if len(upsig) != SIG_LEN:
            n+=1
            continue
        

        noise_add_r = np.random.normal(0, NOISE_LEVEL/30,len(upsig))
        noise_add_complex = np.random.normal(0, NOISE_LEVEL/30,len(upsig))
        noise_add_c = np.zeros(len(noise_add_complex),dtype=np.complex128)

        for i in range(len(noise_add_complex)):
            noise_add_c[i] = complex(0,noise_add_complex[i])


        noisy_upsig = upsig+noise_add_c+noise_add_r
        absupsig = np.array(np.abs(noisy_upsig))  # find the magnitude of the complex signal


        # split into training and test set
        if n < 6333:
            x_train.append(absupsig)
            y_train.append(np.abs(upsig))
        else:
            x_test.append(absupsig)
            y_test.append(np.abs(upsig))

    n+=1
    


# convert to numpy arrays and save as npy files
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
        
np.save(WORKDIR_SAVE + 'x_train_'+str(NOISE_LEVEL)+'.npy', x_train, allow_pickle=True)
np.save(WORKDIR_SAVE + 'y_train_'+str(NOISE_LEVEL)+'.npy', y_train, allow_pickle=True)
np.save(WORKDIR_SAVE + 'x_test_'+str(NOISE_LEVEL)+'.npy', x_test, allow_pickle=True)
np.save(WORKDIR_SAVE + 'y_test_'+str(NOISE_LEVEL)+'.npy', y_test, allow_pickle=True)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)