# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:38:46 2021

reduce temp layer and add prev cracking rates

@author: Shih-Cheng Li
"""
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from cantera import Species, one_atm, Solution, IdealGasReactor, MassFlowController, Reservoir, SolutionArray, MassFlowController, PressureController, ReactorNet
from numpy import zeros, arange, zeros_like, sum, hstack
from argparse import ArgumentParser
from keras import optimizers
from keras import losses
from keras.models import Model
from keras.layers import Input, Dense, Activation, concatenate
from pickle import load
from cantera import add_directory
import sys
from os.path import dirname,join
BASE_DIR=dirname(__file__)
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS

else:
    BASE_DIR = dirname(__file__) 
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
    """
    Module that runs a single PFR Cantera simulation via a series of CSTRs.
    The Plug flow reactor is represented by a linear chain of zero-dimensional
    reactors. The gas at the inlet to the first one has the specified inlet
    composition, and for all others the inlet composition is fixed at the
    composition of the reactor immediately upstream. Since in a PFR model there
    is no diffusion, the upstream reactors are not affected by any downstream
    reactors, and therefore the problem may be solved by simply marching from
    the first to last reactor, integrating each one to steady state.
    Parameters
    =============== =============================================================
    Attribute       Description
    =============== =============================================================
    `reaction_mech` Cantera reaction mechanism (.cti file)
    `T_list`        Temperature profile (°C)
    `pressure_0`    Initial pressue (atm)
    `CCl4_X_0`      Initial CCl4 concentration (mass fraction)
    `mass_flow_rate`Mass flow rate of input gas (T/H)      
    `n_steps`       Number of iterations/number of CSTRs
    `n_pfr`         Number of PFRs
    `length`        Length of each PFR (m)
    `area`          Cross-sectional area (m**2)
    `label`         Label of this mechanism
    =============== =============================================================


    """
    #######################################################################
    # Input Parameters
    #######################################################################
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

    # import the gas model and set the initial conditions
    model = Solution(reaction_mech)
    model.TPX = T_0, pressure_0, composition_0
    dz = length / n_steps
    r_vol = area * dz

    # create a new reactor
    r = IdealGasReactor(model)
    r.volume = r_vol

    # create a reservoir to represent the reactor immediately upstream. Note
    # that the gas object is set already to the state of the upstream reactor
    upstream = Reservoir(model, name='upstream')

    # create a reservoir for the reactor to exhaust into. The composition of
    # this reservoir is irrelevant.
    downstream = Reservoir(model, name='downstream')

    # The mass flow rate into the reactor will be fixed by using a
    # MassFlowController object.
    m = MassFlowController(upstream, r, mdot=mass_flow_rate)

    # We need an outlet to the downstream reservoir. This will determine the
    # pressure in the reactor. The value of K will only affect the transient
    # pressure difference.
    v = PressureController(r, downstream, master=m, K=1e-5)

    sim = ReactorNet([r])

    # define time, space, and other information vectors
    z = (arange(n_steps) + 1) * dz
    t = zeros(n_pfr)  # residence time in each PFR reactor
    # compositions of output stream in each PFR reactor
    compositions = [None] * n_pfr
    states = SolutionArray(r.thermo)

    cracking_rates = [0]
    for i, T in enumerate(T_list[1:]):
        Ti = T_list[i] + 273.15
        Te = T + 273.15
        dT = (Te - Ti) / n_steps
        T = Ti
        t_r = zeros_like(z)  # residence time in each CSTR reactor
        # iterate through the PFR cells
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
    with open(join(BASE_DIR,'clf.pickle'), 'rb') as f:
        scaler = load(f)
    # Load model
    model = build_model()
    model.load_weights(join(BASE_DIR,'model.h5'))

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
            area
        )
        results[label] = {
            'compositions': compositions,
            't': t,
            'cracking_rates': cracking_rates,
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
        rescaled_X_predict = scaler.transform(x_predict)
        x_predict = [rescaled_X_predict[:, 0:2],
                     rescaled_X_predict[:, 2:-1], rescaled_X_predict[:, -1]]
        y = float(model.predict(x_predict))
        prev_y = y
        y_predicted.append(y)
    [print(i * 100, end=',') for i in y_predicted]
    print("\n")


if __name__ == '__main__':
    parser = ArgumentParser(description='model parameters.')
    parser.add_argument('mass_flow_rate', metavar='mass', type=float,
                        help='mass flow rate of inlet stream, unit = T/H')
    parser.add_argument('pressure_0', metavar='Pin',
                        type=float, help='inlet pressure, unit = kg/cm2G')
    parser.add_argument('CCl4_X_0', metavar='CCl4 concentration',
                        type=float, help='inlet CCl4 concentration, unit = ppm')
    parser.add_argument('T_list', metavar='Temperature profile', type=float,
                        nargs='+', help='temperature profile of the process, unit= ℃')
    args = parser.parse_args()
    mass_flow_rate = args.mass_flow_rate
    pressure_0 = args.pressure_0
    CCl4_X_0 = args.CCl4_X_0
    T_list = args.T_list
    n_steps = 100
    n_pfr = len(T_list)-1
    length = 18
    add_directory(join(BASE_DIR))
    if n_pfr == 18:
        area = 3.14 * (186.3 / 1000) ** 2 / 4
    elif n_pfr == 22:
        area = 3.14 * ((262) / 1000) ** 2 / 4
    reaction_mech = {
        'Schirmeister': join(BASE_DIR,'chem_annotated_irreversible.xml')
    }
    # Prediction
    print('Starting prediction...')
    print('cracking rates are:')
    predict(reaction_mech, T_list, pressure_0, CCl4_X_0, mass_flow_rate,
            n_steps, n_pfr, length, area)
