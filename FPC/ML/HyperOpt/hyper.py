

from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials, anneal
from hyperopt.fmin import generate_trials_to_calculate
from numpy.lib.function_base import median
import pandas as pd
import numpy as np
import random as random
from random import choice
import pickle as pk
from pickle import load
from itertools import product
from multiprocessing import Pool
import os
from six import b
from tensorflow.keras.layers import Input, Dense, Activation, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from numpy import zeros, arange, zeros_like, sum, hstack
from cantera import Species, one_atm, Solution, IdealGasReactor, MassFlowController, Reservoir, SolutionArray, MassFlowController, PressureController, ReactorNet
from silence_tensorflow import silence_tensorflow
from sklearn.metrics import mean_absolute_error
from typing import List
import argparse
from scipy.interpolate import CubicSpline


def build_model(lr=0.001):
    first_input = Input(shape=(2,), name='Input_layer_1')
    second_input = Input(shape=(33,), name='Input_layer_2')
    third_input = Input(shape=(1,), name='Prev_cracking')

    layer = Dense(6, name='Hinden_layer_1')(first_input)
    layer = Activation('relu')(layer)

    layer = concatenate([layer, second_input], name='Concatenate_layer')
    layer = Activation('relu')(layer)
    layer = Dense(12, name='Hinden_layer_4')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(12, name='Hinden_layer_5')(layer)
    layer = Activation('relu')(layer)
    layer = concatenate([layer, third_input], name='Concatenate_layer_2')
    layer = Dense(1, name='Hinden_layer_6')(layer)
    output = Activation('sigmoid')(layer)

    model = Model(inputs=[first_input, second_input, third_input],
                  outputs=output)
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=losses.mean_absolute_error,
                  metrics=['accuracy', 'mae'])
    return model


def EDC_cracking(
        reaction_mech,
        T_list,
        pressure_0,
        CCl4_X_0,
        mass_flow_rate,
        n_steps=1000,
        n_pfr=18,
        length=18,
        area=0.03225097679
):
    if CCl4_X_0 > 1:  # ppm
        CCl4_X_0 = float(CCl4_X_0) / 1000000
    T_0 = 273.15 + T_list[0]  # inlet temperature [K]
    pressure_0 *= one_atm
    spcs = Species.listFromFile(reaction_mech)
    for spc in spcs[::-1]:
        if spc.composition == {'C': 2.0, 'Cl': 2.0, 'H': 4.0} and spc.charge == 0:
            EDC_label = spc.name
        if spc.composition == {'C': 1.0, 'Cl': 4.0} and spc.charge == 0:
            CCl4_label = spc.name
    EDC_X_0 = 1 - CCl4_X_0
    composition_0 = '{}:{}, {}:{}'.format(
        EDC_label, EDC_X_0, CCl4_label, CCl4_X_0)
    mass_flow_rate *= 1000 / 3600  # T/H to kg/s
    model = Solution(reaction_mech)
    model.TPX = T_0, pressure_0, composition_0
    dz = length / n_steps
    r_vol = area * dz
    r = IdealGasReactor(model)
    r.volume = r_vol
    upstream = Reservoir(model, name='upstream')
    downstream = Reservoir(model, name='downstream')
    m = MassFlowController(upstream, r, mdot=mass_flow_rate)
    v = PressureController(r, downstream, master=m, K=1e-5)
    sim = ReactorNet([r])

    z = (arange(n_steps) + 1) * dz
    t = zeros(n_pfr)  # residence time in each PFR reactor
    compositions = [None] * n_pfr
    states = SolutionArray(r.thermo)

    cracking_rates = [0]
    for i, T in enumerate(T_list[1:]):
        Ti = T_list[i] + 273.15
        Te = T + 273.15
        dT = (Te - Ti) / n_steps
        T = Ti
        t_r = zeros_like(z)  # residence time in each CSTR reactor
        for n in range(n_steps):
            # simulate the linear T-profile in each reactor
            T = Ti + (n + 1) * dT
            model.TP = T, None
            r.syncState()
            # Set the state of the reservoir to match that of the previous reactor
            model.TPX = r.thermo.TPX
            upstream.syncState()
            # integrate the reactor forward in time until steady state is reached
            sim.reinitialize()
            sim.set_initial_time(0)
            sim.advance_to_steady_state()
            # compute velocity and transform into time
            t_r[n] = r.mass / mass_flow_rate  # residence time in this reactor
            # write output data
            states.append(r.thermo.state)
        t[i] = sum(t_r)
        compositions[i] = model.X[4:]
        cracking_rate = (
            EDC_X_0 - model.X[model.species_index(EDC_label)]) / EDC_X_0
        cracking_rates.append(cracking_rate)
    return compositions, t, cracking_rates


def predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
            n_steps, n_pfr, length, area):
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
    with open('../../results/0430_FPC_modelV6_area/clf.pickle', 'rb') as f:
        scaler = load(f)
    # Load model
    model = build_model()
    model.load_weights('../../results/0430_FPC_modelV6_area/model.h5')

    if type(reaction_mech) != dict:
        raise TypeError('The datatype of `reaction_mech` is {}.It should be a dict.'.format(
            type(reaction_mech)))
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
        x_predict = hstack(x_predict).reshape(1, -1)
        rescaled_X_predict = scaler.transform(x_predict[:, :-1])
        x_predict = [rescaled_X_predict[:, 0:2],
                     rescaled_X_predict[:, 2:], x_predict[:, -1]]
        y = float(model.predict(x_predict))
        prev_y = y
        y_predicted.append(y)
    # [print(f"{(i * 100):.2f}", end=',') for i in y_predicted]
    # print("\n")
    return [i * 100 for i in y_predicted]


def f(T_list=None):
    reaction_mech = {
        'Schirmeister': '../../../KM/2009_Schirmeister_EDC/chem_annotated_irreversible.cti'
    }
    ##TODO##
    head_flux = [18796.31, 21664.45, 22274.44, 23715.35, 24412.25, 24762.11, 25061.39, 24661.78, 23877.03, 21651.58,
                 20088.86, 18665.34, 17595.93, 16724.30, 15850.58, 15106.89, 14309.98, 13538.81, 12903.87, 12209.08, 11632.12, 11115.07]

    # update k and E
    # k_update = [params['r1_k'],params['r2_k'],params['r3_k'],params['r5_k'],params['r9_k'],params['r10_k'],params['r13_k'],params['r15_k'],params['r16_k'],params['r19_k'],params['r22_k'],params['r26_k'],params['r19_k'],params['r19_k']]
    # E_update = [params['r1_E'],params['r2_E'],params['r13_E'],params['r32_E'],params['r33_E'],params['r34_E'],params['r35_E']]
    T1 = T_list[:-1]
    T2 = T_list[1:]
    MlX = predict(reaction_mech, T_list, Pin, 1000, 53,
                  100, len(T_list)-1, 18, 3.14 * (262 / 1000) ** 2 / 4)
    # print(MlX)
    mass_flow_kg = 53053
    Cp = 0.29
    mole_cracking_heat = 171
    T_delta = [T_list[i] - T_list[i-1] for i in range(1,23)]
    X_delta = [MlX[i] - MlX[i-1] for i in range(1,23)]
    Q1 = [mass_flow_kg * Cp * t_delta for t_delta in T_delta]
    Q2 = [mass_flow_kg / 100 *mole_cracking_heat* x_delta for x_delta in X_delta]
    head_flux_mine = [(a+b)/15.94 for a,b in zip(Q1,Q2)]

    hf_delta = mean_absolute_error(head_flux,head_flux_mine)

    loss = abs((MlX[-1]-55)+hf_delta)

    return loss, MlX


def mainf(params):

    raw_T_list = [Tin]
    x = [0, 4, 7, 10, 13, 18, 22]
    # for i in range(1,23):
    #     T_list[i] = T_list[i-1]+params[f"t{i}"]
    for _, value in params.items():
        raw_T_list.append(round((raw_T_list[-1]+value), 2))
    cs = CubicSpline(x, raw_T_list)
    T_list = [cs(i) for i in range(23)]
    loss, MlX = f(T_list=T_list)
    print(f"T_list:{T_list}")
    print(f"loss:{loss}")
    print("final cracking rates:")
    print(f"ML: {MlX[-1]}")

    return {'loss': loss, 'status': STATUS_OK}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters.')
    parser.add_argument('--name', required=True, type=str,
                        help='name of the experiment')
    parser.add_argument('--pin', required=True, type=float)
    parser.add_argument('--tin', required=True, type=float)
    parser.add_argument('--tout', type=float)
    args = parser.parse_args()
    NAME = args.name
    global Pin
    global Tin
    global Tout
    Pin = args.pin
    Tin = args.tin
    Tout = args.tout
    loss_dic = []

    fspace = {'t00': hp.uniform('t00', low=63, high=100),
              't04': hp.uniform('t04', low=32, high=45),
              't07': hp.uniform('t07', low=10, high=25),
              't10': hp.uniform('t10', low=0, high=5),
              't13': hp.uniform('t13', low=0, high=3),
              't18': hp.uniform('t18', low=0, high=3),
              }
    try:
        index_plt = 0
        trials = pk.load(open(f"{NAME}_trials.pk", 'rb'))
    # 初始的數值

    except(FileNotFoundError):
        index_plt = 0
        # [350, 368, 377, 391, 408, 421, 427,
        #  434, 441, 445, 448, 448, 449, 450, 451, 451, 452, 454,
        #               455, 455, 456, 458, 460]
        trials = generate_trials_to_calculate([{
            't00': 65,
            't04': 35,
            't07': 20,
            't10': 3,
            't13': 2,
            't18': 1
        }])
    max_evals = 500
    step = 1
    for i in range(1, max_evals+1, step):
        best = fmin(
            fn=mainf,
            space=fspace,
            algo=anneal.suggest,
            trials=trials,
            max_evals=i,
            rstate=random.seed(42),
            verbose=True
        )

        print("####################################")
        print(best)
        # pk.dump(loss_dic, open(f"{NAME}loss_dic.pk", "wb"))
        pk.dump(trials, open(f"{NAME}_trials.pk", "wb"))
head_flux = [18796.31, 21664.45, 22274.44, 23715.35, 24412.25, 24762.11, 25061.39, 24661.78, 23877.03, 21651.58,
             20088.86, 18665.34, 17595.93, 16724.30, 15850.58, 15106.89, 14309.98, 13538.81, 12903.87, 12209.08, 11632.12, 11115.07]
