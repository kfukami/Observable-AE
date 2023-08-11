# Observable-AE.py
# 2023 Kai Fukami (UCLA, kfukami1@g.ucla.edu)

## Authors:
# Kai Fukami and Kunihiko Taira 
## We provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
#     Ref: K. Fukami and K. Taira,
#     “Grasping extreme aerodynamics on a low-dimensional manifold,”
#     in review, 2023
#
# The code is written for educational clarity and not for speed.
# -- version 1: Aug 11, 2023

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.layers import Input, Add, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, LSTM, Concatenate, Conv2DTranspose
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm as tqdm
from scipy.io import loadmat


#import tensorflow._api.v2.compat.v1 as tf

#tf.disable_v2_behavior()


### CNN-MLP autoencoder with observable augmentation

act = 'tanh'
input_img = Input(shape=(120,240,1))

x1 = Conv2D(32, (3,3),activation=act, padding='same')(input_img)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((5,5),padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = Reshape([12*6*4])(x1)
x1 = Dense(256,activation=act)(x1)
x1 = Dense(128,activation=act)(x1)
x1 = Dense(64,activation=act)(x1)
x1 = Dense(32,activation=act)(x1)

x_lat = Dense(3,activation=act)(x1)

x_CL = Dense(32,activation=act)(x_lat)
x_CL = Dense(64,activation=act)(x_CL)
x_CL = Dense(32,activation=act)(x_CL)
x_CL_final = Dense(1)(x_CL)

x1 = Dense(32,activation=act)(x_lat)
x1 = Dense(64,activation=act)(x1)
x1 = Dense(128,activation=act)(x1)
x1 = Dense(256,activation=act)(x1)
x1 = Dense(288,activation=act)(x1)
x1 = Reshape([6,12,4])(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((5,5))(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x_final = Conv2D(1, (3,3),padding='same')(x1)
autoencoder = Model(input_img, [x_final,x_CL_final])


autoencoder.compile(optimizer='adam', loss='mse',loss_weights=[1,0.05]) # beta = 0.05 determined by L-curve analysis


num_snap = ABC; ## number of training snapshots

y_1 = np.zeros((num_snap,120,240,1)) # vorticity field
y_CL = np.zeros((num_snap,1)) # lift response


from keras.callbacks import ModelCheckpoint,EarlyStopping
X_train, X_test, X_train1, X_test1 = train_test_split(y_1, y_CL, test_size=0.2, random_state=None)
model_cb=ModelCheckpoint('./Model.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=200,verbose=1)
cb = [model_cb, early_cb]
history = autoencoder.fit(X_train,[X_train,X_train1],epochs=50000,batch_size=128,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test, [X_test,X_test1]))
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./History.csv',index=False)


