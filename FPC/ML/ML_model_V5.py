# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:38:46 2021

reduce the layer of temperature

@author: Shih-Cheng Li
"""
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

"""
Changing Log:
    0120: Batch size: 128-->18, 
        Splitting way: all random-> split by a temp-profile a group 

        implemented cross validation 10 folds
        cstr#: 100->50 
        epoch: 10000
"""
######INIIALIZATION#######
######NOT TRUE VALUE ########
# Do k-fold
DOKFOLD = False
TRAINING_DATA = "training"
# global parameters
NAME = "test"  # Name of Temperature and cracking rates curves
dir_path = os.path.dirname(os.path.realpath(__file__))
RESULTPATH = os.path.join(dir_path, 'results', NAME)
PLOTPATH = os.path.join(RESULTPATH, 'plots')
NUMFOLDS = 10
LEARNINGRATE = 0.001
EPOCH = 12000
BATCHSIZE = 18
TRAINING_VERBOSE = 0  # 0: quiet 1: line
CSTR = 100
# Genearalization test
MASS_FLOW_RATE = True
CCL4 = True
INITAL_PRESSURE = True
NUMBER_OF_TUBES = True
PREDICT_ONLY = False
CPU = 8
TRAINING = True
GENE = True
PREDICT = True
T_IN = True
FPC = False
logger = mp.log_to_stderr(logging.DEBUG)
###########################
###########################


def arg_parser():
    parser = argparse.ArgumentParser(description='model parameters.')
    parser.add_argument('--name', required=True, type=str,
                        help='name of the experiment')
    parser.add_argument(
        '--data', default="training_data.csv", help='Training data')
    parser.add_argument('--no_mass', action='store_false',
                        help='don\'t generalization  of mass flow rate')
    parser.add_argument('--no_ccl4', action='store_false',
                        help='don\'t generalization of CCl4')
    parser.add_argument('--no_pressure', action='store_false',
                        help='don\'t generalization of initial pressure')
    parser.add_argument('--no_tubes', action='store_false',
                        help='don\'t generalization of number of tubes')
    parser.add_argument('--kfolds', action='store_true',
                        help='do k-fold cross validation')
    parser.add_argument('--fold_no', default=10,
                        type=int, help='number of folds')
    parser.add_argument('--verbose', default=0, type=int,
                        help='training verbose')
    parser.add_argument('--epoch', default=10000, type=int, help='epoch')
    parser.add_argument('--batch', default=18, type=int, help='Batch size')
    parser.add_argument('--cstr', default=100, type=int,
                        help='Number of iterations/number of CSTRs')
    parser.add_argument(
        '--no_predict', action="store_false", help='no prediction')
    parser.add_argument('--cpu', default=8, type=int, help='ncpus')
    parser.add_argument('--no_train', action='store_false', help='no training')
    parser.add_argument('--no_gene', action='store_false',
                        help='no generalization')
    parser.add_argument('--no_T_in', action='store_false', help='no T_in')
    parser.add_argument('--FPC', action='store_true',
                        help='do everything with FPC data')
    return parser


def set_args(args):

    global DOKFOLD
    global TRAINING_DATA
    global NAME
    global NUMFOLDS
    global RESULTPATH
    global PLOTPATH
    global LEARNINGRATE
    global EPOCH
    global BATCHSIZE
    global TRAINING_VERBOSE
    global CSTR
    global MASS_FLOW_RATE
    global CCL4
    global INITAL_PRESSURE
    global NUMBER_OF_TUBES
    global PREDICT
    global CPU
    global TRAINING
    global GENE
    global T_IN
    global FPC
    # Do k-fold
    DOKFOLD = args.kfolds
    TRAINING_DATA = args.data
    print(TRAINING_DATA)
    # global parameters
    NAME = args.name  # Name of Temperature and cracking rates curves
    dir_path = os.path.dirname(os.path.realpath(__file__))
    RESULTPATH = os.path.join(dir_path, 'results', NAME)
    PLOTPATH = os.path.join(RESULTPATH, 'plots')
    NUMFOLDS = args.fold_no
    LEARNINGRATE = 0.001
    EPOCH = args.epoch
    BATCHSIZE = args.batch
    TRAINING_VERBOSE = args.verbose  # 0: quiet 1: line
    CSTR = args.cstr
    # Genearalization test
    MASS_FLOW_RATE = args.no_mass
    CCL4 = args.no_ccl4
    INITAL_PRESSURE = args.no_pressure
    NUMBER_OF_TUBES = args.no_tubes
    PREDICT = args.no_predict
    CPU = args.cpu
    TRAINING = args.no_train
    GENE = args.no_gene
    T_IN = args.no_T_in
    FPC = args.FPC


# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def build_model(lr=0.001):
    first_input = Input(shape=(2,), name='Input_layer_1')
    second_input = Input(shape=(33,), name='Input_layer_2')

    layer = Dense(6, name='Hinden_layer_1')(first_input)
    layer = Activation('relu')(layer)

    layer = concatenate([layer, second_input], name='Concatenate_layer')
    layer = Activation('relu')(layer)
    layer = Dense(12, name='Hinden_layer_4')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(12, name='Hinden_layer_5')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(1, name='Hinden_layer_6')(layer)
    output = Activation('sigmoid')(layer)

    model = Model(inputs=[first_input, second_input], outputs=output)
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


""" def build_model(lr=0.001):
    first_input = Input(shape=(2,), name='Input_layer_1')
    second_input = Input(shape=(33,), name='Input_layer_2')

    layer = Dense(15, name='Hinden_layer_1')(first_input)
    layer = Activation('relu')(layer)
    layer = Dense(28, name='Hinden_layer_2')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(43, name='Hinden_layer_3')(layer)

    layer = concatenate([layer, second_input], name='Concatenate_layer')
    layer = Activation('relu')(layer)
    layer = Dense(83, name='Hinden_layer_4')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(83, name='Hinden_layer_5')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(1, name='Hinden_layer_6')(layer)
    output = Activation('sigmoid')(layer)

    model = Model(inputs=[first_input, second_input], outputs=output)
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=losses.mean_absolute_error,
                  metrics=['accuracy', 'mae'])
    # model.compile(optimizer=optimizers.Adam(lr=lr),
    #               loss=losses.mean_absolute_error,
    #               metrics=[metrics.MeanAbsoluteError()])

    # model.compile(optimizer=optimizers.Adam(lr=lr),
    #               loss='mse',
    #               metrics=[tf.keras.metrics.MeanSquaredError()])
    return model """


def DataPreprocessor():
    # Load data
    print('Loading data...')
    dataframe = read_csv(TRAINING_DATA)
    array = dataframe.values
    # Separate array into input and output components
    X = array[:, 0:-1]  # X.shape=(alot,35)
    Y = array[:, -1]  # Y.shpae=(alot,1)

    if DOKFOLD:
        scaler = StandardScaler()
        rescaled_X = scaler.fit_transform(X)
        with open(os.path.join(RESULTPATH, 'clf_kFold.pickle'), 'wb') as f:
            pickle.dump(scaler, f)

        Xi, Yi = [], []
        for i in range(len(dataframe)//18):
            Xi.append(rescaled_X[18*i:18*(i+1)])
            Yi.append(Y[18*i:18*(i+1)])

        return Xi, Yi

    else:
        ################################
        # Data spliting to training set and testing set
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, Y, test_size=0.33, random_state=42)
        ################################

        # Test generalization-Data splitting
        # Group data together for each pyrolysis process

        Xi, Yi = [], []
        for i in range(len(dataframe)//18):
            Xi.append(X[18*i:18*(i+1)])
            Yi.append(Y[18*i:18*(i+1)])
        Xi_train, Xi_test, yi_train, yi_test = train_test_split(
            Xi, Yi, test_size=0.20, random_state=42)
        y_train = np.ravel(yi_train)
        y_test = np.ravel(yi_test)
        X_train = np.ravel(Xi_train).reshape(len(y_train), 35)
        X_test = np.ravel(Xi_test).reshape(len(y_test), 35)
        # # Shuffle the lists
        # tmp_train = list(zip(X_train, y_train))
        # random.shuffle(tmp_train)
        # X_train, y_train = zip(*tmp_train)
        # tmp_test = list(zip(X_test, y_test))
        # random.shuffle(tmp_test)
        # X_test, y_test = zip(*tmp_test)
        # X_train, y_train, X_test, y_test = list(X_train), list(y_train), list(X_test), list(y_test)

        # Create the Scaler object
        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        # Fit data on the scaler object
        rescaled_X_train = scaler.fit_transform(X_train)
        # Scale testing data
        rescaled_X_test = scaler.transform(X_test)
        # Save scaler parameter
        with open(os.path.join(RESULTPATH, 'clf.pickle'), 'wb') as f:
            pickle.dump(scaler, f)
        # Change the shape of X in order to separate to ordinary input (in-out temp) and auxilary input (29-species mass fraction and  pressure in ,CCL4 initial ratio,t,tr for every tube)
        X_train = [rescaled_X_train[:, 0:2], rescaled_X_train[:, 2::]]
        X_test = [rescaled_X_test[:, 0:2], rescaled_X_test[:, 2::]]

        return X_train, X_test, y_train, y_test


def train_model(X_train=None, X_test=None, y_train=None, y_test=None, fold_no=None, verbose=1):

    # Set model parameters
    lr = LEARNINGRATE
    epoch = EPOCH
    batch_size = BATCHSIZE
    # Early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=200, restore_best_weights=True)
    csvLogger_callback = CSVLogger('traininglogs.csv')
    # Build model
    print('Bulid model...')
    model = build_model(lr)
    # model.summary()
    plot_model(model, to_file=f"{RESULTPATH}/model.png", show_shapes=True)
    if DOKFOLD:
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
    else:
        print('Training...')

    hist = model.fit(X_train, y_train,
                     batch_size=batch_size,
                     epochs=epoch,
                     callbacks=[early_stopping, csvLogger_callback],
                     validation_split=0.2,
                     verbose=verbose
                     )
    # Save model
    if DOKFOLD:
        model.save_weights(os.path.join(RESULTPATH, str(fold_no)+'_model.h5'))
    else:
        model.save_weights(os.path.join(RESULTPATH, 'model.h5'))
    # learning curve, loss of training and validation sets
    plt.figure()
    ax1 = plt.subplot()
    plt.plot(hist.history['loss'], '#4D80E6', label='loss', linewidth=1.8)
    plt.plot(hist.history['val_loss'], '#FF8033',
             label='val_loss', linewidth=1.8)
    plt.title('Learning curve ')
    plt.ylabel('Mean_absolute_error')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    if DOKFOLD:
        plt.savefig(os.path.join(PLOTPATH, str(fold_no)+'_Learning curve.png'))
        loss, acc = model.evaluate(
            X_test, y_test, batch_size=batch_size, verbose=0)
        print(f'Testset loss for {fold_no}: {loss:.4f}')
        return loss, acc*100

    else:
        loss, acc, *is_anything_else_being_returned = model.evaluate(
            x=X_test, y=y_test, batch_size=batch_size)
        print(is_anything_else_being_returned)
        textstr = f'Testset loss : {loss:4f}'
        print(f'Testset loss/acc:{loss}')
        ax1.text(0.85, 0.55, textstr, ha="right", va="top",
                 transform=ax1.transAxes, bbox=dict(facecolor='wheat', alpha=0.5))
        plt.savefig(os.path.join(PLOTPATH, 'Learning curve.png'))


def predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
            n_steps, n_pfr, length, area, save_fig=False, name='predict', fold_no=None, iter_CCl4=False, FPC=True,print_cracking=False,loss = False,single=False):
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
    # Load scaler parameter
    with open(os.path.join(RESULTPATH, 'clf.pickle'), 'rb') as f:
        scaler = pickle.load(f)
    # Load model
    model = build_model()
    model.load_weights(os.path.join(RESULTPATH, 'model.h5'))
    # if CCl4_X_0 > 1:  # ppm
    #     CCl4_X_0 = float(CCl4_X_0) / 1000000

    if type(reaction_mech) != dict:
        raise TypeError('The datatype of `reaction_mech` is {}.It should be a dict.'.format(
            type(reaction_mech)))
    results = {}
    for label in reaction_mech.keys():
        compositions, t, cracking_rates = EDC_cracking(
            reaction_mech[label],
            T_list,
            pressure_0,
            CCl4_X_0,
            mass_flow_rate,
            n_steps,
            n_pfr,
            length,
            area,
            label=label
        )
        results[label] = {
            'compositions': compositions,
            't': t,
            'cracking_rates': cracking_rates,
        }
    # Use ML model to predict
    KM_label = 'Schirmeister'
    y_predicted = [0]
    for i, T in enumerate(T_list[1:]):
        Ti = T_list[i]
        Te = T
        compositions = results[KM_label]['compositions'][i]
        t = sum(results[KM_label]['t'][:i+1])
        t_r = results[KM_label]['t'][i]
        x_predict = [Ti, Te, compositions, pressure_0, CCl4_X_0, t, t_r]
        x_predict = np.hstack(x_predict).reshape(1, -1)
        rescaled_x_predict = scaler.transform(x_predict)
        x_predict = [rescaled_x_predict[:, 0:2], rescaled_x_predict[:, 2::]]
        y = float(model.predict(x_predict))
        y_predicted.append(y)
    results['ML'] = {'cracking_rates': y_predicted}
    if print_cracking:
        print("ML cracking rates")
        print(results['ML'])
    if loss:
        from sklearn.metrics import mean_absolute_error
        loss = mean_absolute_error(results['Choi']['cracking_rates'],
                                results['ML']['cracking_rates'])
        print(f"loss in {CCl4_X_0}: {loss} ")

    if FPC:
        Ti = T_list[:-1]
        Te = T_list[1:]
        if CCl4_X_0 < 1:  # ppm
            CCl4_X_0 = CCl4_X_0 * 1000000
        df = read_csv("Data/RawDataInput.csv")
        y_ground_df = df.query(
            "CCl4_X_0 == @CCl4_X_0  &"
            "pressure_0 == @pressure_0 &"
            "mass_flow_rate == @mass_flow_rate &"
            "Ti == @Ti &"
            "Te == @Te"
        )['X']
        print(y_ground_df)
        if not y_ground_df.empty:
            y_ground = [0]
            if len(y_ground_df) >= 18:
                for index in y_ground_df.index:
                    try:
                        if y_ground_df.loc[[index+17]].index == index + 17:
                            for i in range(index, index+18):
                                y_ground.append(y_ground_df.loc[i]/100)
                            break
                    except KeyError:
                        print("index + 17 dont exist in y_ground_df, continue")
                        continue

            print(len(y_ground))
            results['FPC'] = {'cracking_rates': y_ground}
            if loss:
                loss = mean_absolute_error(results['FPC']['cracking_rates'],
                                        results['ML']['cracking_rates'])
        else:
            print("FPC data not found")

    if CCl4_X_0 < 1:  # ppm
        CCl4_X_0 = float(CCl4_X_0) * 1000000
    print("Schi cracking rates")
    print([i*100 for i in results['Schirmeister']['cracking_rates']])
    # Plot figure
    if save_fig:
        ndata = len(T_list)
        fig, ax1 = plt.subplots()
        scale = 30
        if area < 3.14 * (200 / 1000) ** 2 / 4:
            scale = 20
        # print("Schi cracking rates")
        # print(results['Schirmeister']*100)

        ln = ax1.plot(range(ndata), T_list, color='r',
                      marker='o', label='Temperature ($^\circ$C)')
        ax1.set_ylabel('Temperature ($^\circ$C)')
        ax1.set_ylim(0, 600)
        # textstr = '\n'.join(
        #     (r'CCl4=%dppm' % (CCl4_X_0),
        #      r'Pin=%.2fkg/cm2G' % (pressure_0),
        #      r'Tin=%d°C' % (T_list[0]),
        #      r'Mass=%dT/H' % (mass_flow_rate),
        #      r'scale=texas'
        #      )
        # )
        textstr = '\n'.join(
            (r'CCl4=%dppm' % (CCl4_X_0),
             r'Pin=%.2fkg/cm2G' % (pressure_0),
             r'Tin=%d°C' % (T_list[0]),
             r'Mass=%dT/H' % (mass_flow_rate),
             r'scale=%d' % (scale)
             )
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top', bbox=props)
        ax2 = ax1.twinx()
        lns = ln
        for label in results.keys():
            cracking_rates = [
                i * 100 for i in results[label]['cracking_rates']]
            lns += ax2.plot(range(ndata), cracking_rates,
                            marker='o', label=label)
        ax2.set_ylabel('Cracking rates (%)')
        ax2.set_ylim(-5, 100)
        text_crack = f"final:{(results['ML']['cracking_rates'][-1]*100):.2f}%"
        fig.text(0.75, 0.75, text_crack, fontsize=9)
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='lower right', frameon=True)

        plt.title('Temperature and cracking rates curves')
        ax1.set_xlabel('PFR index')
        plt.xticks(range(ndata))
        if DOKFOLD:
            plt.savefig(os.path.join(PLOTPATH, f'{fold_no}_{name}.png'))
        elif single:
            if not os.path.exists(os.path.join(PLOTPATH, "predict")):
                os.mkdir(os.path.join(PLOTPATH, "predict"))
            plt.savefig(os.path.join(PLOTPATH,'predict','{}.png'.format(name)))
        elif iter_CCl4 is True:
            if not os.path.exists(os.path.join(PLOTPATH, "predict")):
                os.mkdir(os.path.join(PLOTPATH, "predict"))
            if not os.path.exists(os.path.join(PLOTPATH, "predict", name)):
                os.mkdir(os.path.join(PLOTPATH, "predict", name))

            plt.savefig(os.path.join(
                PLOTPATH, f'predict/{name}/CCl4_{CCl4_X_0:.6f}_mass_{mass_flow_rate}_temp_{T_list[0]}_{name}.png'))
            with open(os.path.join(PLOTPATH, f'predict/{name}/Temperature_profile.txt'), 'w') as f:
                f.write(str(T_list))

    return loss


def generalization_test(scale=20):

    print('Generalization test...')
    # Test case parameters
    reaction_mech = {
        'Schirmeister': '../KM/2009_Schirmeister_EDC/chem_annotated_irreversible.cti',
        'Choi': '../KM/2001_Choi_EDC/chem_annotated_irreversible.cti',
    }
    pressure_0 = 12.4
    CCl4_X_0 = 1000
    T_list = [320, 348, 374.7, 399.3, 424, 451.3, 460.7, 466, 471.3,
              476.7, 478.7, 479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486.7]
    area = 3.14 * (186.3 / 1000) ** 2 / 4
    if FPC:
        mass_flow_rate = 36
    else:
        mass_flow_rate = 72

    if scale == 30:
        T_list = [320, 348, 374.7, 399.3, 424, 451.3, 460.7, 462, 466, 468, 471.3, 474.3,
                  476.7, 477.7, 478.7, 479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486.7]
        area = 3.14 * ((262) / 1000) ** 2 / 4
        if FPC:
            mass_flow_rate = 53
        else:
            mass_flow_rate = 105

    #
    # T_list = [330, 357.3, 382, 406.6, 431.3, 455.1, 461.5, 467.8, 473.1,
    #           476.8, 478.8, 479.4, 480.1, 481.4, 482.4, 483.4, 484.4, 485.4, 486.7]

    # mass_flow_rate = 105
    
    n_pfr = len(T_list)-1
    n_steps = CSTR
    length = 18

    ################################
    # Case1
    if MASS_FLOW_RATE:
        InitialTime = time.time()
        print('Case1: Mass flow rate...')
        # mass_flow_rate_list = np.linspace(45, 82, 100)
        if FPC:
            mass_flow_rate_list = [27, 32, 36]
        else:
            mass_flow_rate_list = [54, 64, 72]
         
        result_1 = {}
        # multiprocessing

        global MFR_generalization

        def MFR_generalization(mfr):
            tup = ()
            loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mfr,
                           n_steps, n_pfr, length, area,loss=True)
            tup = (mfr, loss)
            return tup
        result_1 = {}
        with mp.Pool(CPU) as pool:
            total = pool.map_async(
                MFR_generalization, list(mass_flow_rate_list))
            pool.close()
            pool.join()
            logger.info(total)
            result_1 = dict(total.get())
        # pool = mp.Pool(CPU)
        # total = pool.map_async(MFR_generalization,list(mass_flow_rate_list))
        # pool.close()
        # pool.join()
        # logger.info(total)
        # result_1=dict(total.get())
        for key, value in result_1.items():
            print(f"Error in {key} mass flow rate is {value:.6f} ")

        # for i, mfr in enumerate(mass_flow_rate_list):
        #     loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mfr,
        #                 n_steps, n_pfr, length, area)
        #     # print('Index:{:<4d} Loss:{:<8.5f}'.format(i+1, loss))
        #     result_1[mfr] = loss
        plt.figure()
        plt.title('Generalization-Mass flow rate')
        if FPC:
            plt.axvline(x=27, color='b')
            plt.axvline(x=36, color='b')
            plt.xlim(20, 40)
        else:
            plt.axvline(x=54, color='b')
            plt.axvline(x=72, color='b')
            plt.xlim(45, 80)
        plt.xlabel('Mass flow rate (T/H)')
        plt.ylabel('Mean absolute error')
        plt.scatter(result_1.keys(), result_1.values())
        plt.savefig(os.path.join(
            PLOTPATH, 'Generalization-Mass flow rate0.png'))
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('Mass flow rate generalization takes {} min {} sec.'.format(
            ComputationTime/60, ComputationTime % 60))

    ################################

    # Case2
    if CCL4:
        InitialTime = time.time()
        print('Case2: CCl4 concentration...')
        CCl4_X_0_list = np.linspace(0, 2500, 21)
        result_2 = {}

        # multiprocessing
        global CCl4_generalization

        def CCl4_generalization(CCl4):
            tup = ()
            loss = predict(reaction_mech, T_list, pressure_0, CCl4, mass_flow_rate,
                           n_steps, n_pfr, length, area,loss=True)
            if CCl4 > 1:
                tup = (CCl4, loss)
            else:
                tup = (CCl4*1e6, loss)
            return tup

        pool = mp.Pool(CPU)
        total = pool.map_async(CCl4_generalization, list(CCl4_X_0_list))
        pool.close()
        pool.join()
        logger.info(total)
        result_2 = dict(total.get())
        for key, value in result_2.items():
            print(f"Error in concentration in {key}ppm is {value:.6f} ")
        # for i, CCl4 in enumerate(CCl4_X_0_list):
        #     loss = predict(reaction_mech, T_list, pressure_0, CCl4, mass_flow_rate,
        #                    n_steps, n_pfr, length, area)
        #     # print('Index:{:<4d} Loss:{:<8.5f}'.format(i+1, loss))
        #     result_2[CCl4*1e6] = loss
        plt.figure()
        plt.title('Generalization-Initial inlet CCl4 concentration')
        plt.axvline(x=2500, color='b')
        plt.xlim(-100, 5100)
        plt.xlabel('CCl4 concentration (ppm)')
        plt.ylabel('Mean absolute error')
        plt.scatter(result_2.keys(), result_2.values())
        plt.savefig(os.path.join(
            PLOTPATH, 'Generalization-Initial inlet CCl4 concentration.png'))
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('CCl4 concentration generalization takes {} min {} sec.'.format(
            ComputationTime/60, ComputationTime % 60))

    ################################
    # Case3
    if INITAL_PRESSURE:
        InitialTime = time.time()
        print('Case3: Initial pressure...')
        pressure_0_list = np.linspace(10.4, 14.4, 5)
        result_3 = {}
        # multiprocessing

        global pressure_generalization

        def pressure_generalization(P):
            tup = ()
            loss = predict(reaction_mech, T_list, P, CCl4_X_0, mass_flow_rate,
                           n_steps, n_pfr, length, area,loss=True)
            tup = (P, loss)
            return tup

        pool = mp.Pool(CPU)
        total = pool.map_async(pressure_generalization, list(pressure_0_list))
        pool.close()
        pool.join()
        logger.info(total)
        result_3 = dict(total.get())
        for key, value in result_3.items():
            print(f"Error in {key}atm is {value:.6f} ")

        # for i, P in enumerate(pressure_0_list):
        #     loss = predict(reaction_mech, T_list, P, CCl4_X_0, mass_flow_rate,
        #                 n_steps, n_pfr, length, area)
        #     # print('Index:{:<4d} Loss:{:<8.5f}'.format(i+1, loss))
        #     result_3[P] = loss
        plt.figure()
        plt.title('Generalization-Initial inlet pressure')
        plt.axvline(x=11.4, color='b')
        plt.axvline(x=13, color='b')
        plt.xlim(9.9, 14.5)
        plt.xlabel('Initial pressure (atm)')
        plt.ylabel('Mean absolute error')
        plt.scatter(result_3.keys(), result_3.values())
        plt.savefig(os.path.join(
            PLOTPATH, 'Generalization-Initial inlet pressure0.png'))
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('Initial pressure generalization takes {} min {} sec.'.format(
            ComputationTime/60, ComputationTime % 60))

    ################################

    # Case4
    if NUMBER_OF_TUBES:
        InitialTime = time.time()
        print('Case4: Number of tubes...')
        T_list_dict = {}
        for i in range(14, 23):
            T_list_dict[i] = []
            for j in range(100):
                while True:
                    T_listi = [322, 486]
                    for k in range(i-1):
                        T_listi.append(random.uniform(322, 486))
                    T_listi.sort()
                    if all(T_listi >= np.linspace(322, 486, i+1)):
                        T_list_dict[i].append(T_listi)
                        break
        result_4 = {}
        for i in T_list_dict.keys():
            # print('Number of tubes: {}'.format(i))
            loss_list = []
            result_4[i] = {}

            # multiprocessing

            global tubes_generalization

            def tubes_generalization(T):
                tup = ()
                lossi_list = []
                lossi = predict(reaction_mech, T, pressure_0, CCl4_X_0, mass_flow_rate,
                                n_steps, i, length, area,loss=True)
                lossi_list.append(lossi)
                lossi_list = np.array(lossi_list)
                return lossi_list

            pool = mp.Pool(CPU)
            total = pool.map_async(tubes_generalization, list(T_list_dict[i]))
            pool.close()
            pool.join()
            logger.info(total)
            loss_list = np.array(total.get())
            loss_list = loss_list.ravel()
            mean = np.mean(loss_list)
            std = np.std(loss_list)
            result_4[i]['loss'] = mean
            result_4[i]['std'] = std

            # for j, T in enumerate(T_list_dict[i]):
            #     lossi = predict(reaction_mech, T, pressure_0, CCl4_X_0, mass_flow_rate,
            #                     n_steps, i, length, area)
            #     # print('Index:{:<4d} Loss:{:<8.5f}'.format(j+1, lossi))
            #     loss_list.append(lossi)
            # mean = np.mean(loss_list)
            # std = np.std(loss_list)
            # result_4[i]['loss'] = mean
            # result_4[i]['std'] = std
        plt.figure()
        plt.title('Generalization-Number of tubes')
        plt.xlim(13.5, 23.5)
        plt.xlabel('Number of tubes')
        plt.ylabel('Mean absolute error')
        x = result_4.keys()
        y = [result_4[i]['loss'] for i in result_4.keys()]
        yerr = [result_4[i]['std'] for i in result_4.keys()]
        plt.errorbar(x, y, yerr, fmt="o", ecolor='r',
                     color='b', elinewidth=2, capsize=4)
        plt.xticks(range(14, 23))
        plt.savefig(os.path.join(
            PLOTPATH, 'Generalization-Number of tubes0.png'))
        print('Figures saved in {}'.format(PLOTPATH))
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('Number of tubes generalization takes {} min {} sec.'.format(
            ComputationTime/60, ComputationTime % 60))
    # case 5
    if T_IN:
        InitialTime = time.time()
        
        print('Case5: Initial T_in...')
        T_biglist = [
            [300, 332, 361, 387, 411, 429.6, 443.4, 453.7, 460.5, 464.9,
             468.7, 473, 477.8, 480.5, 481.5, 482.5, 483.5, 484.5, 486.7],
            [310, 340, 368.4, 393.7, 416.5, 441.6, 452, 458.3, 464.6, 471,
             474, 476.5, 478.8, 481.1, 482.1, 483.1, 484.1, 485.1, 486.7],
            [320, 348, 374.7, 399.3, 424, 451.3, 460.7, 466, 471.3, 476.7,
             478.7, 479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486.7],
            [330, 357.3, 382, 406.6, 431.3, 455.1, 461.5, 467.8, 473.1, 476.8,
             478.8, 479.4, 480.1, 481.4, 482.4, 483.4, 484.4, 485.4, 486.7],
            [340, 365.3, 390, 414.6, 439.3, 458.6, 464, 469.3, 474.6, 478,
             479, 479.6, 480.3, 481.6, 482.6, 483.6, 484.6, 485.6, 486.7],
            [350, 374.7, 399.3, 424, 449.3, 460.7, 466, 471.3, 476.7, 478.7,
             479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486, 486.7]
        ]
        if scale == 30:
            T_biglist = [
                [300, 332, 361, 387, 411, 429.6, 443.4, 453.7, 460.5, 464.9, 466.8,
                    468.7, 470.3, 473, 475.7, 477.8, 479.5, 480.5, 481.5, 482.5, 483.5, 484.5, 486.7],
                [310, 340, 368.4, 393.7, 416.5, 441.6, 452, 458.3, 464.6, 469.3, 471.5,
                    473, 475.5, 476.5, 477.5, 478.8, 479.5, 481.1, 482.1, 483.1, 484.1, 485.1, 486.7],
                [320, 356, 380, 406, 430, 457, 466, 472, 474, 476,
                    478, 480, 482, 482, 482, 483, 483, 483, 484, 484, 484],
                [330, 357.3, 382, 406.6, 431.3, 455.1, 461.5, 467.8, 473.1, 476.8,
                    478.8, 479.4, 480.1, 481.4, 482.4, 483.4, 484.4, 485.4, 486.7],
                [340, 365, 390, 416, 440, 467, 476, 477, 478, 479,
                    480, 480, 480, 481, 481, 481, 482, 482, 482, 482, 482],
                [350, 375, 399, 424, 451, 461, 466, 471, 477, 479, 479,
                    479, 480, 480, 480, 480, 480, 480, 480, 480, 481, 481, 481]
            ]

        result_5 = {}
        # multiprocessing

        global T_in_generalization

        def T_in_generalization(T_list):
            tup = ()
            loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
                           n_steps, n_pfr, length, area,save_fig=True, name=T_list[0],loss=True)
            tup = (T_list[0], loss)
            return tup

        pool = mp.Pool(CPU)
        total = pool.map_async(T_in_generalization, T_biglist)
        pool.close()
        pool.join()
        logger.info(total)
        result_5 = dict(total.get())
        for key, value in result_5.items():
            print(f"Error in {key}°C is {value:.6f} ")
        # for i, T_list in enumerate(T_biglist):
        #     loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
        #                 n_steps, n_pfr, length, area)
        #     print('Index:{:<4d} Loss:{:<8.5f}'.format(i+1, loss))
        #     result_5[T_list[0]] = loss
        plt.figure()
        plt.title('Generalization-Initial inlet temperature')
        plt.axvline(x=298, color='b')
        plt.axvline(x=352, color='b')
        plt.xlim(280, 360)
        plt.xlabel('Initial temperature (°C)')
        plt.ylabel('Mean absolute error')
        plt.scatter(result_5.keys(), result_5.values())
        plt.savefig(os.path.join(
            PLOTPATH, 'Generalization-Initial inlet temperature.png'))
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('Initial temperature generalization takes {} min {} sec.'.format(
            ComputationTime/60, ComputationTime % 60))


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    set_args(args)
    np.random.seed(0)
    if not os.path.exists(RESULTPATH):
        os.mkdir(RESULTPATH)
    if not os.path.exists(PLOTPATH):
        os.mkdir(PLOTPATH)

    with open(os.path.join(RESULTPATH, "Parameters.log"), 'w') as f:
        f.write(
            f"\
            Experiment name: {NAME}\n\
            Training Data: {TRAINING_DATA}\n\
            Do k-fold: {DOKFOLD}\n\
            Folds num: {NUMFOLDS}\n\
            Learning rate: {LEARNINGRATE}\n\
            Batch size: {BATCHSIZE}\n\
            Epoch: {EPOCH}\n\
            CSTR num: {CSTR}\n\
            Training vebose: {TRAINING_VERBOSE}\n\
            #Genearalization test\n\
            mass flow rates = {MASS_FLOW_RATE}\n\
            CCl4 = {CCL4}\n\
            inital pressure = {INITAL_PRESSURE}\n\
            number of tubes = {NUMBER_OF_TUBES}\n\
            ncpus = {CPU}\n\
        ")
    print(f"\
            Experiment name: {NAME}\n\
            Training Data: {TRAINING_DATA}\n\
            Do k-fold: {DOKFOLD}\n\
            Folds num: {NUMFOLDS}\n\
            Learning rate: {LEARNINGRATE}\n\
            Batch size: {BATCHSIZE}\n\
            Epoch: {EPOCH}\n\
            CSTR num: {CSTR}\n\
            Training vebose: {TRAINING_VERBOSE}\n\
            #Genearalization test\n\
            mass flow rates = {MASS_FLOW_RATE}\n\
            CCl4 = {CCL4}\n\
            inital pressure = {INITAL_PRESSURE}\n\
            number of tubes = {NUMBER_OF_TUBES}\n\
            T_in = {T_IN}\n\
            ncpus = {CPU}\n\
            training = {TRAINING}\n\
            prediction only = {PREDICT_ONLY}\n\
                ")

    # Do k fold
    if DOKFOLD:
        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []
        # Normalize X and Y
        Xi, yi = DataPreprocessor()

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=NUMFOLDS, shuffle=True)

        # Measure time
        InitialTime = time.time()

        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train, test in kfold.split(Xi, yi):
            # Change the shape of X in order to separate to ordinary input (in-out temp)
            # and auxilary input (29-species mass fraction and  pressure in
            # ,CCL4 initial ratio,t,tr for every tube)
            y = np.ravel(yi)
            X = np.ravel(Xi).reshape(len(y), 35)
            X_train = X[train]
            X_test = X[test]
            new_X_train = [X_train[:, 0:2], X_train[:, 2::]]
            new_X_test = [X_test[:, 0:2], X_test[:, 2::]]
            loss, acc_percent = train_model(
                new_X_train, new_X_test, y[train], y[test], fold_no=fold_no, verbose=TRAINING_VERBOSE)
            acc_per_fold.append(acc_percent)
            loss_per_fold.append(loss)

            # Increase fold number
            fold_no += 1
        # Measure time
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('Training takes {} min {} sec.'.format(
            ComputationTime/60, ComputationTime % 60))

        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print(
                '------------------------------------------------------------------------')
            print(
                f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(
            f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

    else:

        X_train, X_test, y_train, y_test = DataPreprocessor()
        # Model training

        if TRAINING:
            InitialTime = time.time()
            train_model(X_train, X_test, y_train, y_test)
            FinalTime = time.time()
            ComputationTime = FinalTime-InitialTime
            print('Training takes {} min {} sec.'.format(
                ComputationTime/60, ComputationTime % 60))
            time.sleep(20)
        # Generalization test
        if GENE:
            generalization_test(scale=20)
        # Set parameter (Test case1)
        reaction_mech = {
            'Schirmeister': '../KM/2009_Schirmeister_EDC/chem_annotated_irreversible.cti',
            'Choi': '../KM/2001_Choi_EDC/chem_annotated_irreversible.cti',
        }

        if PREDICT:
            ##20##
            # T_list = [310, 340, 360, 386, 430, 460, 464, 467, 469,
            #           470, 472, 473, 475, 477, 479, 482, 484, 485, 486]
            # T_list = [350, 380, 396, 413, 428, 436, 444, 453, 459,
            #   463, 465, 467, 469, 470, 471, 472, 473, 474, 475]
            # T_list = [350, 374.7, 399.3, 424, 449.3, 460.7, 466, 471.3, 476.7, 478.7,
            #           479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486, 486.7]
            # T_list = [320, 350, 375, 399, 424, 440, 445, 450, 452,
            #           456, 460, 462, 464, 467, 470, 474, 475, 476, 477]
            # T_list = [320, 348, 374.7, 399.3, 424, 451.3, 460.7, 466, 471.3,
            #   476.7, 478.7, 479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486.7]
            # T_list = [330, 357.3, 382, 406.6, 431.3, 455.1, 461.5, 467.8, 473.1, 476.8,
            #  478.8, 479.4, 480.1, 481.4, 482.4, 483.4, 484.4, 485.4, 486.7]
            # T_list = [350, 374.7, 399.3, 424, 449.3, 460.7, 466, 471.3, 476.7,
            #           478.7, 479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486, 486.7]
            # T_list = [320, 350, 375, 399, 424, 440, 445, 450, 452,
            #           456, 460, 462, 464, 467, 470, 474, 475, 476, 477]
            # T_list = [300, 332, 361, 387, 411, 429.6, 443.4, 453.7, 460.5,
            #           464.9, 468.7, 473, 477.8, 480.5, 481.5, 482.5, 483.5, 484.5, 486.7]
            # T_list = [340, 365.3, 390, 414.6, 439.3, 458.6, 464, 469.3, 474.6,
            #           478, 479, 479.6, 480.3, 481.6, 482.6, 483.6, 484.6, 485.6, 486.7]

            ##30##
            # T_list = [320, 348, 374.7, 399.3, 424, 451.3, 460.7, 462, 466, 468, 471.3, 474.3,
                    #   476.7, 477.7, 478.7, 479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486.7]
            # T_list = [300, 332, 361, 387, 411, 429.6, 443.4, 453.7, 455.1, 457.5, 460.5,
                    #   464.9, 466.8, 468.7, 473, 475.4, 477.8, 480.5, 481.5, 482.5, 483.5, 484.5, 486.7]
            # T_list = [350, 374.7, 399.3, 424, 449.3, 460.7, 466, 468.5, 471.3, 473.5, 475.8,
            #           476.7, 477.7, 478.7, 479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486, 486.7]
            # T_list = [350, 374.7, 399.3, 424, 440.3, 443.7, 446, 448.5, 451.3, 453.5, 455.8,
            #           456.7, 457.7, 458.7, 459.3, 460, 461.3, 462.3, 463.3, 464.3, 465.3, 466, 466.7]
            # T_list = [350, 374.7, 399.3, 424, 440.3, 450.7, 456, 458.5, 461.3, 463.5, 465.8,
            #           466.7, 467.7, 468.7, 469.3, 470, 471.3, 472.3, 473.3, 474.3, 475.3, 476, 476.7]
            T_list = [320, 350, 375, 399, 424, 440, 445, 450, 452, 454, 456,
                      458, 460, 461, 462, 464, 467, 470, 472, 474, 476, 476, 477]
            # T_list = [322, 350, 375, 399, 424, 451, 455, 460, 464, 467, 469,
            #   470, 474, 476, 477, 478, 479, 480, 481, 483, 484, 485, 486]
            ##random##
            # T_list = [322, 350, 375, 399, 424, 441, 455, 462, 466, 470, 472,
            #           474, 475, 476, 477, 478, 479, 480, 481, 483, 484, 485, 486]
            # T_list = [350, 375, 399, 424, 451, 461, 466, 471, 477, 479, 479,
            #           479, 480, 480, 480, 480, 480, 480, 480, 480, 481, 481, 481]
            # T_list = [350, 374.7, 399.3, 424, 451.3, 460.7, 466, 471.3, 476.7, 478.7, 479.3,
            #           479.4, 479.5, 479.6, 479.7, 479.8, 479.9, 480, 480.2, 480.4, 480.6, 480.8, 482]
            # T_list = [300, 340, 375, 399, 424, 451, 455, 460, 463, 465, 468,
            #           471, 473, 475, 477, 479, 480, 481, 482, 483, 484, 485, 486]
            # T_list = [347, 358, 383, 413, 438, 454, 456, 458, 460, 462, 464,
                    #   466, 468, 470, 472, 474, 476, 477, 477, 478, 479, 480, 481]

            pressure_0 = 12.4
            CCl4_X_0 = 0
            mass_flow_rate = 53
            n_steps = CSTR
            n_pfr = len(T_list)-1
            length = 18
            # n_pfr = 22
            # area = 3.14 * (186.3 / 1000) ** 2 / 4
            # area = 3.14 * (238.76 / 1000) ** 2 / 4

            if n_pfr == 18:
                area = 3.14 * (186.3 / 1000) ** 2 / 4
            elif n_pfr == 22:
                area = 3.14 * ((262) / 1000) ** 2 / 4
            # area = 3.14 * ((9.4 * 2.54)/100) ** 2 /4
            name = '30_53_320_h476'
            single= False
            # Prediction
            print('Starting prediction...')
            if single:
                loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
                            n_steps, n_pfr, length, area, save_fig=True, name=name,print_cracking= False,FPC=FPC,single=single)
            else:
                for CCl4_X_0 in np.linspace(0, 2500, 21):
                    loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
                                n_steps, n_pfr, length, area, save_fig=True, name=name, iter_CCl4=True, FPC=FPC)
                from matplotlib.animation import FuncAnimation
                import imageio
                import glob

                filenames = glob.glob(f"{PLOTPATH}/predict/{name}/*_{name}.png")
                print(filenames)
                with imageio.get_writer(f'{PLOTPATH}/predict/{name}/{name}.gif', mode='I') as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)

                print('Mean absolute error of `{}` is {:.5f}'.format(NAME, loss))
