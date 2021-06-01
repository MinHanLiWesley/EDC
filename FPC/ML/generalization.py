import os
import pickle
import random
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import time
import random
import argparse
import pdb
import multiprocessing as mp
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from keras.layers import Input, Dense, Activation, concatenate
from keras.models import Model
from keras import losses, metrics
from keras import optimizers
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf

from createTrainingData import EDC_cracking


def build_model(lr=0.001):
    first_input = Input(shape=(2,), name='Input_layer_1')
    second_input = Input(shape=(33,), name='Input_layer_2')
    third_input = Input(shape=(1,), name='Prev_cracking')

    layer = Dense(11, name='Hinden_layer_1')(first_input)
    layer = Activation('relu')(layer)

    layer = concatenate([layer, second_input], name='Concatenate_layer')
    layer = Activation('relu')(layer)
    layer = Dense(8, name='Hinden_layer_2')(layer)
    layer = Activation('relu')(layer)
    # layer = Dense(6, name='Hinden_layer_4')(layer)
    # layer = Activation('relu')(layer)
    # layer = Dense(9, name='Hinden_layer_5')(layer)
    # layer = Activation('relu')(layer)
    layer = concatenate([layer, third_input], name='Concatenate_layer_2')
    layer = Dense(1, name='Hinden_layer_3')(layer)
    output = Activation('sigmoid')(layer)

    model = Model(inputs=[first_input, second_input, third_input],
                  outputs=output)
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=losses.mean_absolute_error,
                  metrics=['accuracy', 'mae'])
    # model.compile(optimizer=optimizers.Adam(lr=lr),
    #               loss=losses.mean_absolute_error,
    #               metrics=[metrics.MeanAbsoluteError()])

    # model.compile(optimizer=optimizers.Adam(lr=lr),
    #               loss='mse',
    #               metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

df = pd.read_csv("training_data_FPC_V8_addprev_area.csv")
array = df.values
X = array[:, 0:-1]
Y = array[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=777)
scaler = StandardScaler()
rescaled_X_train = scaler.fit_transform(X_train[:, :-1])
rescaled_X_test = scaler.transform(X_test[:, :-1])
with open(('clf.pickle'), 'wb') as f:
        pickle.dump(scaler, f)
X_train=[rescaled_X_train[:, 0:2],
                rescaled_X_train[:, 2:], X_train[:, -1]]
X_test = [rescaled_X_test[:, 0:2],
                rescaled_X_test[:, 2:], X_test[:, -1]]

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=200, restore_best_weights=True)
model = build_model()
hist = model.fit(X_train, y_train,
                batch_size=128,
                epochs=100,
                callbacks=[early_stopping],
                validation_split=0.2
                )

# model.save_weights("model2.h5")
a,b,_ = model.evaluate(x =X_test,y= y_test)
print(a)