# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:34:12 2021
ML2 use old data
@author: Shih-Cheng Li
"""
import numpy as np
import json
import traceback
import time
from pandas import read_csv
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dill
from keras.layers import Input, Dense, Activation, Dropout, concatenate
from keras.models import Model
from keras import losses, metrics
from keras import optimizers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials, anneal
from hyperopt.fmin import generate_trials_to_calculate
# import ray
# from ray import tune
# from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.suggest.hyperopt import HyperOptSearch
# from ray.tune.suggest import ConcurrencyLimiter
from hyperopt import hp
from tensorflow.python.keras.utils import layer_utils
from createTrainingData import EDC_cracking
import tensorflow.keras as keras
from sklearn.metrics import mean_absolute_error


# class TuneReporterCallback(keras.callbacks.Callback):
#     """
#     Tune Callback for Keras.

#     The callback is invoked every epoch.
#     """

#     def __init__(self, logs={}):
#         self.iteration = 0
#         super(TuneReporterCallback, self).__init__()

#     def on_epoch_end(self, batch, logs={}):
#         self.iteration += 1
#         tune.report(keras_info=logs,
#                     val_loss=logs['val_loss'], mean_loss=logs.get("loss"))

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['Trained_Model']


def get_data():
    dataframe = read_csv('../training_data_FPC_V8_addprev_area.csv')
    # dataframe = read_csv('../training_data_FPC_V7_addprev_Temprand.csv')
    array = dataframe.values
    # Separate array into input and output components
    X = array[:, 0:-1]
    Y = array[:, -1]
    # Data spliting to training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=777)
    # Create the Scaler object
    scaler = StandardScaler()
    # Fit data on the scaler object
    rescaled_X_train = scaler.fit_transform(X_train[:, :-1])
    # Scale testing data
    rescaled_X_test = scaler.transform(X_test[:, :-1])
    # Change the shape of X
    x_train = [rescaled_X_train[:, 0:2],
               rescaled_X_train[:, 2:], X_train[:, -1]]
    x_test = [rescaled_X_test[:, 0:2], rescaled_X_test[:, 2::], X_test[:, -1]]
    return x_train, y_train, x_test, y_test, scaler
'''
def create_model(params):
    first_input = Input(shape=(2,),)
    second_input = Input(shape=(33,))
    third_input = Input(shape=(1,), name='Prev_cracking')

    # layer = Dense(int(params['units1-1']))(first_input)
    # layer = Activation('relu')(layer)

    layer = Dense(int(params['units1-2']))(first_input)
    layer = Activation('relu')(layer)

    if params['choice1']['layers'] == 'three':
        layer = Dense(int(params['choice1']['units1-3']))(layer)
        layer = Activation('relu')(layer)
    elif params['choice1']['layers'] == 'four':
        layer = Dense(int(params['choice1']['units1--3']))(layer)
        layer = Activation('relu')(layer)
        layer = Dense(int(params['choice1']['units1-4']))(layer)
        layer = Activation('relu')(layer)

    layer = concatenate([layer, second_input])
    layer = Activation('relu')(layer)

    # layer = Dense(int(params['units2-1']))(layer)
    # layer = Activation('relu')(layer)

    layer = Dense(int(params['units2-2']))(layer)
    layer = Activation('relu')(layer)

    if params['choice2']['layers'] == 'three':
        layer = Dense(int(params['choice2']['units2-3']))(layer)
        layer = Activation('relu')(layer)
    elif params['choice2']['layers'] == 'four':
        layer = Dense(int(params['choice2']['units2--3']))(layer)
        layer = Activation('relu')(layer)
        layer = Dense(int(params['choice2']['units2-4']))(layer)
        layer = Activation('relu')(layer)
    layer = concatenate([layer, third_input])
    layer = Dense(1)(layer)
    output = Activation('sigmoid')(layer)

    model = Model(inputs=[first_input, second_input,
                  third_input], outputs=output)

    model.compile(optimizer=optimizers.Adam(lr=params['lr']),
                  loss=losses.mean_absolute_error,
                  metrics=[metrics.MeanAbsoluteError()])
    return model

'''

def create_model(params):
    first_input = Input(shape=(2,),)
    second_input = Input(shape=(33,))
    third_input = Input(shape=(1,), name='Prev_cracking')

    # layer = Dense(int(params['units1-1']))(first_input)
    # layer = Activation('relu')(layer)

    layer = Dense(int(params['units1-2']))(first_input)
    layer = Activation('relu')(layer)

    if params['choice1']['layers'] == 'three':
        layer = Dense(int(params['choice1']['units1-3']))(layer)
        layer = Activation('relu')(layer)
    elif params['choice1']['layers'] == 'four':
        layer = Dense(int(params['choice1']['units1--3']))(layer)
        layer = Activation('relu')(layer)
        layer = Dense(int(params['choice1']['units1-4']))(layer)
        layer = Activation('relu')(layer)

    layer = concatenate([layer, second_input])
    layer = Activation('relu')(layer)

    # layer = Dense(int(params['units2-1']))(layer)
    # layer = Activation('relu')(layer)

    layer = Dense(int(params['units2-2']))(layer)
    layer = Activation('relu')(layer)

    if params['choice2']['layers'] == 'three':
        layer = Dense(int(params['choice2']['units2-3']))(layer)
        layer = Activation('relu')(layer)
    elif params['choice2']['layers'] == 'four':
        layer = Dense(int(params['choice2']['units2--3']))(layer)
        layer = Activation('relu')(layer)
        layer = Dense(int(params['choice2']['units2-4']))(layer)
        layer = Activation('relu')(layer)
    layer = concatenate([layer, third_input])
    layer = Dense(1)(layer)
    output = Activation('sigmoid')(layer)

    model = Model(inputs=[first_input, second_input,
                  third_input], outputs=output)

    model.compile(optimizer=optimizers.Adam(lr=params['lr']),
                  loss=losses.mean_absolute_error,
                  metrics=[metrics.MeanAbsoluteError()])
    return model


def predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
            n_steps, n_pfr, length, area, scaler, model):
    """
    Load the saved parameters of StandardScaler() and rebuild the ML model to
    do predictions.

    =============== =============================================================
    Attribute       Description
    =============== =============================================================
    `reaction_mech` Doctinary of Cantera reaction mechanism(s) (.cti file)
    `T_list`        Temperature profile (°C)
    `pressure_0`    Initial pressue (atm)
    `CCl4_X_0`      Initial CCl4 concentration (mass fraction)
    `mass_flow_rate`Mass flow rate of input gas (T/H)
    `n_steps`       Number of iterations/number of CSTRs
    `n_pfr`         Number of PFRs
    `length`        Length of each PFR (m)
    `area`          Cross-sectional area (m**2)
    `save_fig`      Save figure to `plots` folder
    `name`          The file name of the saving figure
    =============== =============================================================


    """

    results = {}
    for label in reaction_mech.keys():
        compositions, t, __ = EDC_cracking(
            reaction_mech[label],
            T_list,
            pressure_0,
            CCl4_X_0,
            mass_flow_rate,
            n_steps,
            n_pfr,
            length,
            area
        )
        results[label] = {
            'compositions': compositions,
            't': t
        }
    # Use ML model to predict
    KM_label = 'Schirmeister'
    y_predicted = [0]
    prev_y = 0
    for i, T in enumerate(T_list[1:]):
        Ti = T_list[i]
        Te = T
        compositions = results[KM_label]['compositions'][i]
        t = sum(results[KM_label]['t'][:i+1])
        t_r = results[KM_label]['t'][i]

        x_predict = [Ti, Te, compositions,
                     pressure_0, CCl4_X_0, t, t_r, prev_y]
        x_predict = np.hstack(x_predict).reshape(1, -1)
        rescaled_X_predict = scaler.transform(x_predict[:,:-1])
        x_predict = [rescaled_X_predict[:, 0:2],
                     rescaled_X_predict[:, 2:], x_predict[:, -1]]
        y = float(model.predict(x_predict))
        prev_y = y
        y_predicted.append(y)
    # [print(f"{(i * 100):.2f}", end=',') for i in y_predicted]
    # print("\n")
    return [i * 100 for i in y_predicted]


def tune_model(config):
    print(config)
    x_train, y_train, x_test, y_test, scalar = get_data()
    model = create_model(config)
    checkpoint_callback = ModelCheckpoint(
        "model4.h5", monitor='loss', save_best_only=True)

    # Enable Tune to make intermediate decisions by using a Tune Callback hook. This is Keras specific.
    callbacks = [checkpoint_callback]

    # Train the model
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        verbose=0,
        batch_size=int(config['batch_size']),
        epochs=int(config['epochs']),
        callbacks=callbacks)

    loss, acc, *is_anything_else_being_returned = model.evaluate(
        x=x_test, y=y_test, batch_size=int(config['batch_size']))
    T_list = [348.3, 368.3, 388.3, 405.8, 421.8, 435.3, 446.3, 455.3, 462.3, 467.7, 470.7,
              472.7, 474.1, 475.3, 476.5, 477.6, 478.7, 479.7, 480.6, 481.5, 482.3, 483.1, 483.9]
    texas_X = [0., 0.52, 1.36, 2.64, 4.42, 6.76, 9.60, 12.84, 16.36, 20.02, 23.67,
               27.19, 30.55, 33.73, 36.75, 39.62, 42.34, 44.93, 47.39, 49.72, 51.94, 54.05, 56.05]
    reaction_mech = {
        'Schirmeister': '../../KM/2009_Schirmeister_EDC/chem_annotated_irreversible.cti'
    }
    mine_X = predict(reaction_mech, T_list, 10.33, 0, 48.15, 100, len(
        T_list)-1, 16.3, 3.14 * (238.76 / 1000) ** 2 / 4, scalar, model)
    texas_loss = mean_absolute_error(texas_X, mine_X)
    print(f"ML: {mine_X}")
    print(f"val:{loss}")
    print(f"Mine texas %: {mine_X[-1]}")
    print(f"texas_loss = {texas_loss}")
    return {'loss': (loss + texas_loss), 'status': STATUS_OK,'Trained_Model': model.get_weights()}


# This seeds the hyperparameter sampling.
np.random.seed(5)
# hyperparameter_space = {'choice1': hp.choice('num_layers1',
#                                              [{'layers':'two', },
#                                                  {'layers': 'three',
#                                                   'units1-3': hp.quniform('units1-3', 3, 50, 1)},
#                                                  {'layers': 'four',
#                                                   'units1--3': hp.quniform('units1--3', 3, 50, 1),
#                                                   'units1-4': hp.quniform('units1-4', 3, 50, 1)}
#                                              ]),

#                         'choice2': hp.choice('num_layers2',
#                                              [{'layers': 'two', },
#                                               {'layers': 'three',
#                                                  'units2-3': hp.quniform('units2-3', 2, 50, 1)},
#                                                  {'layers': 'four',
#                                                   'units2--3': hp.quniform('units2--3', 2, 50, 1),
#                                                   'units2-4': hp.quniform('units2-4', 2, 50, 1)}
#                                               ]),
#                         'units2-2': hp.quniform('units2-2', 5, 50, 1),
#                         'units1-2': hp.quniform('units1-2', 5, 50, 1),

#                         'batch_size': 128,

#                         'epochs': hp.quniform('epochs', 500, 1500, 1),
#                         # 'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
#                         'lr': 0.001,
#                         }
hyperparameter_space = {'choice1': hp.choice('num_layers1',
                                             [{'layers':'two', },
                                                 {'layers': 'three',
                                                  'units1-3': hp.quniform('units1-3', 3, 20, 1)},
                                                 {'layers': 'four',
                                                  'units1--3': hp.quniform('units1--3', 3, 20, 1),
                                                  'units1-4': hp.quniform('units1-4', 3, 20, 1)}
                                             ]),

                        'choice2': hp.choice('num_layers2',
                                             [{'layers': 'two', },
                                              {'layers': 'three',
                                                 'units2-3': hp.quniform('units2-3', 2, 20, 1)},
                                                 {'layers': 'four',
                                                  'units2--3': hp.quniform('units2--3', 2, 20, 1),
                                                  'units2-4': hp.quniform('units2-4', 2, 20, 1)}
                                              ]),
                        'units2-2': hp.quniform('units2-2', 5, 20, 1),
                        'units1-2': hp.quniform('units1-2', 5, 20, 1),

                        'batch_size': hp.quniform('batch_size', 62, 128, 1),

                        'epochs': hp.quniform('epochs', 500, 1500, 1),
                        # 'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
                        'lr': 0.001,
                        }
# intial_best_config = [{'num_layers1': 1,
#                        'units1-3': 6,
#                        'num_layers2': 0,
#                        'units2-2': 6,
#                        'batch_size': 128,
#                        'epochs': 501,
#                        'lr': 0.001}]
try:
    index_plt = 0
    trials = dill.load(open("ML/mlhp4.pk", 'rb'))
# 初始的數值

except(FileNotFoundError):
    index_plt = 0
    # [350, 368, 377, 391, 408, 421, 427,
    #  434, 441, 445, 448, 448, 449, 450, 451, 451, 452, 454,
    #               455, 455, 456, 458, 460]
    trials = Trials()

import random
loss_dic = []
max_evals = 100
step = 1
for i in range(1, max_evals+1, step):
    best = fmin(
        fn=tune_model,
        space=hyperparameter_space,
        algo=anneal.suggest,
        trials=trials,
        max_evals=i,
        rstate=random.seed(42),
        verbose=True
    )
    

    print("####################################")
    print(best)
    dill.dump(loss_dic, open("ML/mlhp_loss4.pk", "wb"))
    dill.dump(trials, open("ML/mlhp4.pk", "wb"))

model_wei= getBestModelfromTrials(trials)
dill.dump(model_wei,open("ML/bestmodel4.h5", "wb"))
# print(config)