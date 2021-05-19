# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:38:46 2021

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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from keras.layers import Input, Dense, Activation, concatenate
from keras.models import Model
from keras import losses, metrics
from keras import optimizers
from keras import callbacks
from keras.utils import plot_model

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
#Do k-fold
DOKFOLD= False
TRAINING_DATA="training"
# global parameters
NAME = "test" # Name of Temperature and cracking rates curves
dir_path = os.path.dirname(os.path.realpath(__file__))
RESULTPATH = os.path.join(dir_path,'results',NAME)
PLOTPATH=os.path.join(RESULTPATH,'plots')
NUMFOLDS=10
LEARNINGRATE=0.001
EPOCH=12000
BATCHSIZE=18
TRAINING_VERBOSE=0 #0: quiet 1: line
CSTR = 100
#Genearalization test
MASS_FLOW_RATE = True
CCL4 = True
INITAL_PRESSURE = True
NUMBER_OF_TUBES = True
PREDICT_ONLY = False
CPU = 8
TRAINING = True
logger = mp.log_to_stderr(logging.DEBUG)
###########################
###########################
def arg_parser():
    parser = argparse.ArgumentParser(description='model parameters.')
    parser.add_argument('--name',required=True,type=str,help='name of the experiment')
    parser.add_argument('--data',default="training_data.csv",help='Training data')
    parser.add_argument('--no_mass',action='store_false',help='do generalization  of mass flow rate')
    parser.add_argument('--no_ccl4',action='store_false',help='do generalization of CCl4')
    parser.add_argument('--no_pressure',action='store_false',help='do generalization of initial pressure')
    parser.add_argument('--no_tubes',action='store_false',help='do generalization of number of tubes')
    parser.add_argument('--kfolds',action='store_true',help='do k-fold cross validation')
    parser.add_argument('--fold_no',default=10,type=int,help='number of folds')
    parser.add_argument('--verbose',default=0,type=int,help='training verbose')
    parser.add_argument('--epoch',default=10000,type=int,help='epoch')
    parser.add_argument('--batch',default=18,type=int,help='Batch size')
    parser.add_argument('--cstr',default=100,type=int,help='Number of iterations/number of CSTRs')
    parser.add_argument('--predict_only',action="store_true",help='do prediction only')
    parser.add_argument('--cpu',default=8,type=int,help='ncpus')
    parser.add_argument('--no_train',action='store_false',help='no training')
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
    global PREDICT_ONLY
    global CPU
    global TRAINING
    #Do k-fold
    DOKFOLD= args.kfolds
    TRAINING_DATA=args.data
    print(TRAINING_DATA)
    # global parameters
    NAME = args.name # Name of Temperature and cracking rates curves
    dir_path = os.path.dirname(os.path.realpath(__file__))
    RESULTPATH = os.path.join(dir_path,'results',NAME)
    PLOTPATH=os.path.join(RESULTPATH,'plots')
    NUMFOLDS=args.fold_no
    LEARNINGRATE=0.001
    EPOCH=args.epoch
    BATCHSIZE=args.batch
    TRAINING_VERBOSE=args.verbose #0: quiet 1: line
    CSTR = args.cstr
    #Genearalization test
    MASS_FLOW_RATE = args.no_mass
    CCL4 = args.no_ccl4
    INITAL_PRESSURE = args.no_pressure
    NUMBER_OF_TUBES = args.no_tubes
    PREDICT_ONLY = args.predict_only
    CPU = args.cpu
    TRAINING = args.no_train

def build_model(lr=0.001):
    first_input = Input(shape=(2,), name='Input_layer_1')
    second_input = Input(shape=(32,), name='Input_layer_2')
    
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
                  metrics=[metrics.MeanAbsoluteError()])
    return model
     
def DataPreprocessor():
    # Load data
    print('Loading data...')
    dataframe = read_csv(TRAINING_DATA)
    array = dataframe.values
    # Separate array into input and output components
    X = array[:,0:-1] # X.shape=(alot,35)
    Y = array[:,-1]  # Y.shpae=(alot,1)

    if DOKFOLD:
        scaler = StandardScaler()
        rescaled_X = scaler.fit_transform(X)
        with open(os.path.join(RESULTPATH,'clf_kFold.pickle'), 'wb') as f:
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
            Xi, Yi, test_size=0.33, random_state=42)
        y_train = np.ravel(yi_train)
        y_test = np.ravel(yi_test)
        X_train = np.ravel(Xi_train).reshape(len(y_train),35)
        X_train = np.concatenate((X_train[:,:-3],X_train[:,-2:]),axis=1)
        X_test = np.ravel(Xi_test).reshape(len(y_test),35)
        X_test = np.concatenate((X_test[:,:-3],X_test[:,-2:]),axis=1)
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
        # Fit data on the scaler object
        rescaled_X_train = scaler.fit_transform(X_train)
        # Scale testing data
        rescaled_X_test = scaler.transform(X_test)
        # Save scaler parameter
        with open(os.path.join(RESULTPATH,'clf.pickle'), 'wb') as f:
            pickle.dump(scaler, f)
        # Change the shape of X in order to separate to ordinary input (in-out temp) and auxilary input (29-species mass fraction and  pressure in ,CCL4 initial ratio,t,tr for every tube)
        X_train = [rescaled_X_train[:,0:2], rescaled_X_train[:,2:]]
        X_test = [rescaled_X_test[:,0:2], rescaled_X_test[:,2:]]

        return X_train,X_test,y_train,y_test
    
    


def train_model(X_train=None,X_test=None,y_train=None,y_test=None,fold_no=None,verbose=0):

    
    # Set model parameters
    lr = LEARNINGRATE
    epoch = EPOCH
    batch_size = BATCHSIZE 
    # Early stopping
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=100)
    # Build model
    print('Bulid model...')
    model = build_model(lr)
    # model.summary()
    # plot_model(model, show_shapes=True)
    if DOKFOLD:
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
    else:
        print('Training...')

    hist = model.fit(X_train, y_train,
                     batch_size = batch_size,
                     epochs = epoch,
                     callbacks = [early_stopping],
                     validation_split = 0.2,
                     verbose = verbose)
    # Save model
    if DOKFOLD:
        model.save_weights(os.path.join(RESULTPATH,str(fold_no)+'_model.h5'))
    else:
        model.save_weights(os.path.join(RESULTPATH,'model.h5'))
    # learning curve, loss of training and validation sets
    plt.figure()
    plt.plot(hist.history['loss'],'#4D80E6',label='loss',linewidth=1.8)
    plt.plot(hist.history['val_loss'],'#FF8033',label='val_loss',linewidth=1.8)
    plt.title('Learning curve ')
    plt.ylabel('Mean_absolute_error')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    if DOKFOLD:
        plt.savefig(os.path.join(PLOTPATH,str(fold_no)+'_Learning curve.png'))
        loss, acc = model.evaluate(X_test , y_test, batch_size=batch_size,verbose=0)
        print(f'Testset loss for {fold_no}: {loss:.4f}')
        return loss, acc*100

    else:
        plt.savefig(os.path.join(PLOTPATH,'Learning curve.png'))
        loss, acc = model.evaluate(X_test , y_test, batch_size=batch_size)
        print('Testset loss : %.4f' % (loss))
    
    
    
    

def predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate, 
            n_steps, n_pfr, length, area, save_fig=False, name='predict',fold_no=None,iter_CCl4=False):
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
    model.load_weights(os.path.join(RESULTPATH,'model.h5'))
    
    if type(reaction_mech) != dict:
        raise TypeError('The datatype of `reaction_mech` is {}.It should be a dict.'.format(type(reaction_mech)))
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
        x_predict = [Ti, Te, compositions, pressure_0, t, t_r]
        x_predict = np.hstack(x_predict).reshape(1, -1)
        rescaled_x_predict = scaler.transform(x_predict)
        x_predict = [rescaled_x_predict[:,0:2],rescaled_x_predict[:,2:]]
        y = float(model.predict(x_predict))
        y_predicted.append(y)
    results['ML'] = {'cracking_rates': y_predicted}
    from sklearn.metrics import mean_absolute_error
    loss = mean_absolute_error(results['Choi']['cracking_rates'],
                               results['ML']['cracking_rates'])

    # Plot figure
    if save_fig:
        ndata = len(T_list)
        fig, ax1 = plt.subplots()
        
        ln = ax1.plot(range(ndata), T_list, color='r', marker='o', label='Temperature ($^\circ$C)')
        ax1.set_ylabel('Temperature ($^\circ$C)')
        ax1.set_ylim(0, 600)
        
        ax2 = ax1.twinx()
        lns = ln
        for label in results.keys():
            cracking_rates = [i * 100 for i in results[label]['cracking_rates']]
            lns += ax2.plot(range(ndata), cracking_rates, marker='o', label=label)   
        ax2.set_ylabel('Cracking rates (%)')
        ax2.set_ylim(-5, 100)
    
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='lower right', frameon=True)
        
        plt.title('Temperature and cracking rates curves')
        ax1.set_xlabel('PFR index')
        plt.xticks(range(ndata))
        if DOKFOLD:
            plt.savefig(os.path.join(PLOTPATH, f'{fold_no}_{name}.png'))
        elif iter_CCl4 is True:
            plt.savefig(os.path.join(PLOTPATH, f'CCl4_{CCl4_X_0:.6f}_{name}.png'))
        else:
            plt.savefig(os.path.join(PLOTPATH, '{}.png'.format(name)))
    
    return loss
    


def generalization_test():
    
    print('Generalization test...')
    # Test case parameters
    reaction_mech = {
        'Schirmeister': '../KM/2009_Schirmeister_EDC/chem_annotated_irreversible.cti',
        'Choi': '../KM/2001_Choi_EDC/chem_annotated_reversible.cti',
        }
    T_list = [322, 350, 375, 399, 424, 451, 461, 466, 471, 477, 479, 479, 480, 481, 482, 483, 484, 485, 486]
    pressure_0 = 11.4
    CCl4_X_0 = 0.001
    mass_flow_rate = 72
    n_steps = CSTR
    n_pfr = 18
    length = 18
    area = 3.14 * ((8 * 2.54 - 2 * 0.818) / 100) ** 2 / 4
    
    ################################
    # Case1
    if MASS_FLOW_RATE:
        InitialTime = time.time()
        print('Case1: Mass flow rate...')
        mass_flow_rate_list = np.linspace(45, 82, 100)
        result_1 = {}
        # multiprocessing

        global MFR_generalization
        def MFR_generalization(mfr):
            tup = ()
            loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mfr,
                        n_steps, n_pfr, length, area)
            tup = (mfr,loss)
            return tup
        
        pool = mp.Pool(CPU)
        total = pool.map(MFR_generalization,list(mass_flow_rate_list))
        pool.close()
        logger.info(total)
        result_1=dict(total)
        for key,value in result_1.items():
            print(f"Error in {key} mass flow rate is {value:.6f} ")

        # for i, mfr in enumerate(mass_flow_rate_list):
        #     loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mfr,
        #                 n_steps, n_pfr, length, area)
        #     # print('Index:{:<4d} Loss:{:<8.5f}'.format(i+1, loss))
        #     result_1[mfr] = loss
        plt.figure()
        plt.title('Generalization-Mass flow rate')
        plt.axvline(x=55, color = 'b')
        plt.axvline(x=72, color = 'b')
        plt.xlim(44.5, 82.5)
        plt.xlabel('Mass flow rate (T/H)')
        plt.ylabel('Mean absolute error')
        plt.scatter(result_1.keys(), result_1.values())
        plt.savefig(os.path.join(PLOTPATH, 'Generalization-Mass flow rate.png'))
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('Mass flow rate generalization takes {} min {} sec.'.format(ComputationTime/60,ComputationTime%60))

    ################################
    
    # Case2
    if CCL4:
        InitialTime = time.time()
        print('Case2: CCl4 concentration...')
        CCl4_X_0_list = np.linspace(0, 0.0025, 100) 
        result_2 = {}

        # multiprocessing
        global CCl4_generalization
        def CCl4_generalization(CCl4):
            tup = ()
            loss = predict(reaction_mech, T_list, pressure_0, CCl4, mass_flow_rate,
                        n_steps, n_pfr, length, area)
            tup = (CCl4*1e6,loss)
            return tup
        
        pool = mp.Pool(CPU)
        total = pool.map(CCl4_generalization,list(CCl4_X_0_list))
        pool.close()
        logger.info(total)
        result_2=dict(total)
        for key,value in result_2.items():
            print(f"Error in concentration in {key}ppm is {value:.6f} ")
        # for i, CCl4 in enumerate(CCl4_X_0_list):
        #     loss = predict(reaction_mech, T_list, pressure_0, CCl4, mass_flow_rate,
        #                    n_steps, n_pfr, length, area)
        #     # print('Index:{:<4d} Loss:{:<8.5f}'.format(i+1, loss))
        #     result_2[CCl4*1e6] = loss
        plt.figure()
        plt.title('Generalization-Initial inlet CCl4 concentration')
        plt.axvline(x=2500, color = 'b')
        plt.xlim(-100, 5100)
        plt.xlabel('CCl4 concentration (ppm)')
        plt.ylabel('Mean absolute error')
        plt.scatter(result_2.keys(), result_2.values())
        plt.savefig(os.path.join(PLOTPATH, 'Generalization-Initial inlet CCl4 concentration.png'))
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('CCl4 concentration generalization takes {} min {} sec.'.format(ComputationTime/60,ComputationTime%60))
    
    ################################
    # Case3
    if INITAL_PRESSURE:
        InitialTime = time.time()
        print('Case3: Initial pressure...')
        pressure_0_list = np.linspace(10, 14.4, 100)
        result_3 = {}
        # multiprocessing

        global pressure_generalization
        def pressure_generalization(P):
            tup = ()
            loss = predict(reaction_mech, T_list, P, CCl4_X_0, mass_flow_rate,
                        n_steps, n_pfr, length, area)
            tup = (P,loss)
            return tup
        
        pool = mp.Pool(CPU)
        total = pool.map(pressure_generalization,list(pressure_0_list))
        pool.close()

        logger.info(total)
        result_3=dict(total)
        for key,value in result_3.items():
            print(f"Error in {key}atm is {value:.6f} ")

        # for i, P in enumerate(pressure_0_list):
        #     loss = predict(reaction_mech, T_list, P, CCl4_X_0, mass_flow_rate,
        #                 n_steps, n_pfr, length, area)
        #     # print('Index:{:<4d} Loss:{:<8.5f}'.format(i+1, loss))
        #     result_3[P] = loss
        plt.figure()
        plt.title('Generalization-Initial inlet pressure')
        plt.axvline(x=11.4, color = 'b')
        plt.axvline(x=13, color = 'b')
        plt.xlim(9.9, 14.5)
        plt.xlabel('Initial pressure (atm)')
        plt.ylabel('Mean absolute error')
        plt.scatter(result_3.keys(), result_3.values())
        plt.savefig(os.path.join(PLOTPATH, 'Generalization-Initial inlet pressure.png'))
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('Initial pressure generalization takes {} min {} sec.'.format(ComputationTime/60,ComputationTime%60))
    
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
                lossi_list =[]
                lossi = predict(reaction_mech, T, pressure_0, CCl4_X_0, mass_flow_rate,
                                n_steps, i, length, area)
                lossi_list.append(lossi)
                lossi_list = np.array(lossi_list)
                return lossi_list
            
            pool = mp.Pool(CPU)
            total = pool.map(tubes_generalization,list(T_list_dict[i]))
            pool.close()
            logger.info(total)
            loss_list = np.array(total)
            loss_list = loss_list.ravel()
            mean = np.mean(loss_list)
            std = np.std(loss_list)
            result_4[i]['loss']= mean
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
        plt.errorbar(x, y, yerr, fmt="o", ecolor='r', color='b', elinewidth=2, capsize=4)
        plt.xticks(range(14, 23))
        plt.savefig(os.path.join(PLOTPATH, 'Generalization-Number of tubes.png'))
        print('Figures saved in {}'.format(PLOTPATH))
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('Number of tubes generalization takes {} min {} sec.'.format(ComputationTime/60,ComputationTime%60))
    
if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    set_args(args)
    np.random.seed(0)
    if not os.path.exists(RESULTPATH):
        os.mkdir(RESULTPATH)
    if not os.path.exists(PLOTPATH):
        os.mkdir(PLOTPATH)
    
    with open(os.path.join(RESULTPATH,"Parameters.log"),'w') as f:
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
            ncpus = {CPU}\n\
            training = {TRAINING}\n\
            prediction only = {PREDICT_ONLY}\n\
                ")
        
    #Do k fold
    if DOKFOLD:
        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []
        # Normalize X and Y
        Xi, yi = DataPreprocessor()

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=NUMFOLDS,shuffle=True)

        # Measure time
        InitialTime = time.time()

        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train,test in kfold.split(Xi,yi):
            # Change the shape of X in order to separate to ordinary input (in-out temp) and auxilary input (29-species mass fraction and  pressure in ,CCL4 initial ratio,t,tr for every tube)
            y = np.ravel(yi)
            X = np.ravel(Xi).reshape(len(y),35)
            X_train = X[train]
            X_test = X[test]
            new_X_train = [X_train[:,0:2], X_train[:,2::]]
            new_X_test = [X_test[:,0:2], X_test[:,2::]]
            loss, acc_percent =train_model(new_X_train,new_X_test,y[train],y[test],fold_no=fold_no,verbose=TRAINING_VERBOSE)
            acc_per_fold.append(acc_percent)
            loss_per_fold.append(loss)

            # Increase fold number
            fold_no += 1
        # Measure time
        FinalTime = time.time()
        ComputationTime = FinalTime-InitialTime
        print('Training takes {} min {} sec.'.format(ComputationTime/60,ComputationTime%60))

        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')



        

    
    else:

        X_train,X_test,y_train,y_test = DataPreprocessor()
        #Model training
        if not PREDICT_ONLY:
            
            if TRAINING:
                InitialTime = time.time()
                train_model(X_train,X_test,y_train,y_test)
                FinalTime = time.time()
                ComputationTime = FinalTime-InitialTime
                print('Training takes {} min {} sec.'.format(ComputationTime/60,ComputationTime%60))

            # Generalization test
            generalization_test()    
        # Set parameter (Test case1)
        reaction_mech = {
            'Schirmeister': '../KM/2009_Schirmeister_EDC/chem_annotated_irreversible.cti',
            'Choi': '../KM/2001_Choi_EDC/chem_annotated_reversible.cti',
            }
        T_list = [322, 350, 375, 399, 424, 451, 461, 466, 471, 477, 479, 479, 480, 481, 482, 483, 484, 485, 486]
        pressure_0 = 11.4
        CCl4_X_0 = 0.0003
        mass_flow_rate = 72
        n_steps = CSTR
        n_pfr = 18
        length = 18
        area = 3.14 * ((8 * 2.54 - 2 * 0.818)/ 100) ** 2 / 4 
        # Prediction
        print('Starting prediction...')
        # loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
        #             n_steps, n_pfr, length, area, save_fig=True, name=NAME)
        for CCl4_X_0 in np.linspace(0, 0.0003, 20):
            loss = predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
                    n_steps, n_pfr, length, area, save_fig=True, name=NAME,iter_CCl4 = True)

        print('Mean absolute error of `{}` is {:.5f}'.format(NAME, loss))
