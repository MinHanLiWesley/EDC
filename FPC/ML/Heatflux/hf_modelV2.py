

from scipy.optimize import Bounds, LinearConstraint, minimize, minimize_scalar,fmin_powell,fmin,basinhopping
from math import pi
import cantera as ct
import numpy as np
import pickle
from keras.layers import Input, Dense, Activation, concatenate
from keras.models import Model
from keras import losses, metrics
from keras import optimizers
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.utils import plot_model
from keras import backend as K
from scipy.optimize.optimize import fminbound
import tensorflow as tf
import itertools
NAME='0606_FPC_modelV11'
def objective_function(delta_T, hf_obj, prev_y,raw_T_list,mfr=53.053,p=13,ccl4=1000,
                       mole_cracking_heat=171, Cp=0.29, Area=pi*11.1*2.54*18/100):                       
    print(f"delta_T:{delta_T}")
    print(f"hf_onj:{hf_obj}")
    T_list = raw_T_list + [raw_T_list[-1]+delta_T]
    print(f"Tubes {len(T_list)-1}:")
    print(T_list)
    compos,t,t_sum = EDC_cracking(T_list,p,ccl4,mfr,n_pfr=len(T_list)-1)
    with open(f'../results/{NAME}/clf.pickle', 'rb') as f:
        scaler = pickle.load(f)
    # Load model
    model = build_model()
    model.load_weights(f"../results/{NAME}/model.h5")
    x_predict=np.hstack([T_list[-2], T_list[-1], compos,
                     p, ccl4, t, t_sum, prev_y]).reshape(1,-1)
    rescaled_X_predict = scaler.transform(x_predict[:, :-1])
    x_predict = [rescaled_X_predict[:, 0:2],
                    rescaled_X_predict[:, 2:], x_predict[:, -1]]
    X = float(model.predict(x_predict))
    print(f"X:{X}")
    mfr *= 1000  # T/H to kg/H
    EDC_cracked = (X-prev_y)*mfr  # already / 100
    
    Q1 = mfr * Cp * delta_T
    print(f"Q1:{Q1}")
    if X > prev_y:
        Q2 = EDC_cracked * mole_cracking_heat
    else:
        Q2 = 100000000
    print(f"Q2:{Q2}")
    hf = (Q1+Q2)/Area  # surface area
    print(f"hf_calculated:{hf}")
    print(f"hf loss: {abs(hf-hf_obj)}")
    return abs(hf-hf_obj)

def build_model(lr=0.001):
    first_input = Input(shape=(2,), name='Input_layer_1')
    second_input = Input(shape=(31,), name='Input_layer_2')
    third_input = Input(shape=(1,), name='Prev_cracking')

    layer = Dense(13, name='Hinden_layer_1')(first_input)
    layer = Activation('relu')(layer)
    layer = Dense(15, name='Hinden_layer_2')(layer)
    layer = Activation('relu')(layer)
    # layer = Dense(16, name='Hinden_layer_3')(layer)
    # layer = Activation('relu')(layer)
    layer = concatenate([layer, second_input], name='Concatenate_layer')
    layer = Activation('relu')(layer)
    # layer = Dense(11, name='Hinden_layer_4')(layer)
    # layer = Activation('relu')(layer)
    layer = Dense(12, name='Hinden_layer_4')(layer)
    layer = Activation('relu')(layer)
    layer = concatenate([layer, third_input], name='Concatenate_layer_2')
    layer = Dense(1, name='Hinden_layer_5')(layer)
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

def EDC_cracking(
        T_list,
        pressure_0,
        CCl4_X_0,
        mass_flow_rate,
        reaction_mech="../../KM/2009_Schirmeister_EDC/chem_annotated_irreversible.cti",
        n_steps=1000,
        n_pfr=22,
        length=18,
        area=3.14 * (262 / 1000) ** 2 / 4
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
    pressure_0 *= ct.one_atm
    spcs = ct.Species.listFromFile(reaction_mech)
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
    model = ct.Solution(reaction_mech)
    model.TPY = T_0, pressure_0, composition_0
    dz = length / n_steps
    r_vol = area * dz

    # create a new reactor
    r = ct.IdealGasReactor(model)
    r.volume = r_vol

    # create a reservoir to represent the reactor immediately upstream. Note
    # that the gas object is set already to the state of the upstream reactor
    upstream = ct.Reservoir(model, name='upstream')

    # create a reservoir for the reactor to exhaust into. The composition of
    # this reservoir is irrelevant.
    downstream = ct.Reservoir(model, name='downstream')

    # The mass flow rate into the reactor will be fixed by using a
    # MassFlowController object.
    m = ct.MassFlowController(upstream, r, mdot=mass_flow_rate)

    # We need an outlet to the downstream reservoir. This will determine the
    # pressure in the reactor. The value of K will only affect the transient
    # pressure difference.
    v = ct.PressureController(r, downstream, master=m, K=1e-5)

    sim = ct.ReactorNet([r])

    # define time, space, and other information vectors
    z = (np.arange(n_steps) + 1) * dz
    t = np.zeros(n_pfr)  # residence time in each PFR reactor
    # compositions of output stream in each PFR reactor
    compositions = [None] * n_pfr
    states = ct.SolutionArray(r.thermo)

    cracking_rates = [0]
    for i, T in enumerate(T_list[1:]):
        Ti = T_list[i] + 273.15
        Te = T + 273.15
        dT = (Te - Ti) / n_steps
        T = Ti
        t_r = np.zeros_like(z)  # residence time in each CSTR reactor
        # iterate through the PFR cells
        for n in range(n_steps):
            # simulate the linear T-profile in each reactor
            T = Ti + (n + 1) * dT
            model.TP = T, None
            r.syncState()
            # Set the state of the reservoir to match that of the previous reactor
            model.TPY = r.thermo.TPY
            upstream.syncState()
            # integrate the reactor forward in time until steady state is reached
            sim.reinitialize()
            sim.set_initial_time(0)
            sim.advance_to_steady_state()
            # compute velocity and transform into time
            t_r[n] = r.mass / mass_flow_rate  # residence time in this reactor
            # write output data
            states.append(r.thermo.state)
        t[i] = np.sum(t_r)
        compositions[i] = model.Y[4:]
        cracking_rate = (
            EDC_X_0 - model.Y[model.species_index(EDC_label)]) / EDC_X_0
        cracking_rates.append(cracking_rate)
        t_total = np.sum(t)
    return compositions[-1], t[-1], t_total


def optimize_hf(Te, Ti=350., target_X=55.0, mfr=53.053, p=13., CCl4=1000,
                mole_cracking_heat=171, Cp=0.29, SurArea=pi*11.1*2.54*18/100):
    Total_hf = (target_X * mole_cracking_heat*mfr*1000/100 + (Te - Ti)*mfr*Cp*1000)/SurArea
    # ratio of per tube to total hf
    ratio = [0.045775715, 0.052760646, 0.054246201, 0.057755337, 0.059452529, 0.060304555, 0.06103342,
             0.060060211, 0.058149083, 0.052729307, 0.048923541, 0.045456762, 0.042852356, 0.04072963,
             0.038601821, 0.036790655, 0.0348499, 0.032971834, 0.031425517, 0.029733461, 0.028328363, 0.027069159]
    reaction_mech = '../../KM/2009_Schirmeister_EDC/test.cti'
    T_list = [Ti]
    X=[0]
    bounds=[[8,25],[12,25],[12,25],[12,25],[10,20],[5,18],[5,18],[5,18],[2,10],[1,8],[1,6],[0,3],[0,3],[0,2],[0,2],[0,1],[0,1],[0,1]]
    for i in range(22):
        # res = fminbound(objective_function,bounds[i][0],bounds[i][1],args=(ratio[i]*Total_hf,X[-1],T_list),
            # disp=3)
        # res = basinhopping(objective_function,bounds[i][0],interval=5)
        res = minimize(objective_function,bounds[i][0],tol=0.001,args=(ratio[i]*Total_hf,X[-1],T_list))
        # res = fmin(objective_function,2,ftol = 500,maxiter=1000,args=(ratio[i]*Total_hf,X[-1],T_list),
        #     disp=True)
        T_list.append(T_list[-1]+res)
        compos,t,t_sum = EDC_cracking(T_list,p,CCl4,mfr,n_pfr=len(T_list)-1)
        with open(f'../results/{NAME}/clf.pickle', 'rb') as f:
            scaler = pickle.load(f)
        # Load model
        model = build_model()
        model.load_weights(f"../results/{NAME}/model.h5")
        x_predict=np.hstack([T_list[-2], T_list[-1], compos,
                        p, CCl4, t, t_sum, X[-1]]).reshape(1,-1)
        rescaled_X_predict = scaler.transform(x_predict[:, :-1])
        x_predict = [rescaled_X_predict[:, 0:2],
                        rescaled_X_predict[:, 2:], x_predict[:, -1]]
        X.append(float(model.predict(x_predict)))

    print(T_list)
    print(X)
        

if __name__ == '__main__':
    optimize_hf(Te=465.)