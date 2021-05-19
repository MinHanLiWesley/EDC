# -*- coding: utf-8 -*-
# from EDC.FPC.ML.ML_model_coke import CHCL3
import time
import random
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import sys
import os
import argparse

from pandas.core.accessor import delegate_names


def EDC_cracking(
        reaction_mech,
        T_list,
        pressure_0,
        CCl4_X_0,
        mass_flow_rate,
        n_steps=1000,
        n_pfr=18,
        length=18,
        area=0.03225097679,
        label=None,
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
    print(f"cracking {CCl4_X_0}")
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
    model.TPX = T_0, pressure_0, composition_0
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
        t[i] = np.sum(t_r)
        compositions[i] = model.X[4:]
        cracking_rate = (
            EDC_X_0 - model.X[model.species_index(EDC_label)]) / EDC_X_0
        cracking_rates.append(cracking_rate)
    return compositions, t, cracking_rates


'''
def EDC_cracking_C2H3Cl3(
        reaction_mech,
        T_list,
        pressure_0,
        CCl4_X_0,
        mass_flow_rate,
        n_steps=1000,
        n_pfr=18,
        length=18,
        area=0.027403710696000005,
        label=None,
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
    T_0 = 273.15 + T_list[0]  # inlet temperature [K]
    pressure_0 *= ct.one_atm
    spcs = ct.Species.listFromFile(reaction_mech)
    for spc in spcs[::-1]:
        if spc.composition == {'C': 2.0, 'Cl': 2.0, 'H': 4.0} and spc.charge == 0:
            EDC_label = spc.name   
        if spc.composition == {'C': 2.0, 'Cl': 3.0, 'H': 3.0} and spc.charge == 0:
            C2H3Cl3_label = spc.name
        if spc.composition == {'C': 1.0, 'Cl': 4.0} and spc.charge == 0:
            CCl4_label = spc.name
    EDC_X_0 = 1 - CCl4_X_0
    composition_0 = '{}:{}, {}:{}'.format(EDC_label, EDC_X_0, CCl4_label, CCl4_X_0)
    mass_flow_rate *= 1000 / 3600  # T/H to kg/s
    
    # import the gas model and set the initial conditions
    model = ct.Solution(reaction_mech)
    model.TPX = T_0, pressure_0, composition_0
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
    compositions = [None] * n_pfr  # compositions of output stream in each PFR reactor
    states = ct.SolutionArray(r.thermo)
    
    cracking_rates = [0]
    forming_rates = [0]
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
        t[i] = np.sum(t_r)
        compositions[i] = model.X[4:]
        cracking_rate = (EDC_X_0 - model.X[model.species_index(EDC_label)]) / EDC_X_0
        forming_rate = model.X[model.species_index(C2H3Cl3_label)]
        forming_rates.append(forming_rate)
        cracking_rates.append(cracking_rate)
    return compositions, t, cracking_rates ,forming_rates
'''


def EDC_coking(
        reaction_mech,
        T_list,
        pressure_0,
        CCl4_X_0,
        CHCl3_X_0,
        Tri_X_0,
        CP_X_0,
        mass_flow_rate,
        n_steps=1000,
        n_pfr=18,
        length=18,
        area=0.03225097679,
        label=None,
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
    if CHCl3_X_0 > 1:
        CHCl3_X_0 = float(CHCl3_X_0) / 1000000
    if Tri_X_0 > 1:
        Tri_X_0 = float(Tri_X_0)/1000000
    if CP_X_0 > 1:
        CP_X_0 = float(CP_X_0)/1000000
    print(
        f"coking CCl4:{CCl4_X_0}, CHCl3:{CHCl3_X_0}, Tri:{Tri_X_0}, CP:{CP_X_0}")
    T_0 = 273.15 + T_list[0]  # inlet temperature [K]
    pressure_0 *= ct.one_atm
    spcs = ct.Species.listFromFile(reaction_mech)
    for spc in spcs[::-1]:
        if spc.composition == {'C': 2.0, 'Cl': 2.0, 'H': 4.0} and spc.charge == 0:
            EDC_label = spc.name
        if spc.composition == {'C': 1.0, 'Cl': 3.0, 'H': 1.0} and spc.charge == 0:
            CHCl3_label = spc.name
        if spc.composition == {'C': 2.0, 'Cl': 3.0, 'H': 3.0} and spc.charge == 0:
            Tri_label = spc.name
        if spc.composition == {'C': 4.0, 'Cl': 1.0, 'H': 5.0} and spc.charge == 0:
            if spc.name != '1-CP(23)':
                CP_label = spc.name
        if spc.composition == {'C': 1.0, 'Cl': 4.0} and spc.charge == 0:
            CCl4_label = spc.name
    EDC_X_0 = 1 - CCl4_X_0 - CP_X_0 - Tri_X_0 - CHCl3_X_0
    composition_0 = '{}:{}, {}:{}, {}:{}, {}:{}, {}:{}'.format(
        EDC_label, EDC_X_0,
        CCl4_label, CCl4_X_0,
        CHCl3_label, CHCl3_X_0,
        Tri_label, Tri_X_0,
        CP_label, CP_X_0
    )
    mass_flow_rate *= 1000 / 3600  # T/H to kg/s

    # import the gas model and set the initial conditions
    model = ct.Solution(reaction_mech)
    model.TPX = T_0, pressure_0, composition_0
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
        t[i] = np.sum(t_r)
        compositions[i] = model.X[4:]
    return compositions, t


def plot(T_list, cracking_rates):

    ndata = len(T_list)
    fig, ax1 = plt.subplots()

    l1 = ax1.plot(range(ndata), T_list, color='r',
                  marker='o', label='Temperature ($^\circ$C)')
    ax1.set_ylabel('Temperature ($^\circ$C)')
    ax1.set_ylim(0, 600)

    ax2 = ax1.twinx()
    cracking_rates = [i * 100 for i in cracking_rates]
    l2 = ax2.plot(range(ndata), cracking_rates, color='b',
                  marker='o', label='Cracking rates (%)')
    ax2.set_ylabel('Cracking rates (%)')
    ax2.set_ylim(-5, 100)

    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')

    plt.title('Temperature and cracking rates curves')
    ax1.set_xlabel('PFR index')
    plt.xticks(range(ndata))
    plt.show()


def test():
    reaction_mech = '../KM/2001_Choi_EDC/chem_annotated_reversible.cti'
    # reaction_mech = './2009_Schirmeister_EDC/chem_annotated_irreversible.cti'
    T_list = [322, 350, 375, 399, 424, 451, 461, 466, 471,
              477, 479, 479, 480, 481, 482, 483, 484, 485, 486]
    pressure_0 = 11.4
    CCl4_X_0 = 0.001
    mass_flow_rate = 72
    n_steps = 100
    n_pfr = 18
    length = 18
    area = 3.14 * ((8 * 2.54 - 2 * 0.818) / 100) ** 2 / 4
    compositions, t, cracking_rates = EDC_cracking(
        reaction_mech,
        T_list,
        pressure_0,
        CCl4_X_0,
        mass_flow_rate,
        n_steps,
        n_pfr,
        length,
        area,
    )
    plot(T_list, cracking_rates)


def main():

    reaction_mech_x = '../KM//2009_Schirmeister_EDC/chem_annotated_irreversible.cti'
    # reaction_mech_y = '../KM/2001_Choi_EDC/chem_annotated_reversible.cti'

    # Cantera simulation parameters
    n = 1
    parser = argparse.ArgumentParser(description='Creating coking data.')
    parser.add_argument('--mass', required=True, type=float)
    parser.add_argument('--tin', required=True, type=float)
    parser.add_argument('--pressure', required=True, type=float)
    parser.add_argument('--ccl4', required=True, type=float)
    parser.add_argument('--chcl3', required=True, type=float)
    parser.add_argument('--tri', required=True, type=float)
    parser.add_argument('--cp', required=True, type=float)
    arg = parser.parse_args()
    CCl4_X_0 = arg.ccl4
    # 50 CCl4
    CHCl3_X_0 = arg.chcl3
    Tri_X_0 = arg.tri
    CP_X_0 = arg.cp
    T_in = arg.tin
    mass_flow_rate = arg.mass
    pressure_0 = arg.pressure
    DATADIR = "../Data/dataV3"
    DATANAME = f'{mass_flow_rate}_{pressure_0}_{CCl4_X_0}_{CHCl3_X_0}_{Tri_X_0}_{CP_X_0}_{T_in}.csv'
    print('Training data creation initiated at {0}'.format(time.asctime()))
    if not os.path.exists(DATADIR):
        os.makedirs(DATADIR)
    if os.path.exists(f'{DATADIR}/{DATANAME}'):
        sys.exit()
    f = open(f'{DATADIR}/{DATANAME}', 'a')
    if os.stat(f'{DATADIR}/{DATANAME}').st_size == 0:
        f.write('Ti,Te,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,pressure_0,CCl4,CHCl3,Tri,CP,t,tr\n')
    T_list = []
    if T_in == 300:
        T_list = [300, 332, 361, 387, 411, 429.6, 443.4, 453.7, 460.5,
                  464.9, 468.7, 473, 477.8, 480.5, 481.5, 482.5, 483.5, 484.5, 486.7]
    elif T_in == 310:
        T_list = [310, 340, 368.35, 393.65, 416.5, 441.6, 452, 458.3, 464.6,
                  471, 474, 476.5, 478.8, 481.1, 482.1, 483.1, 484.1, 485.1, 486.7]
    elif T_in == 320:
        T_list = [320, 348, 374.7, 399.3, 424, 451.3, 460.7, 466, 471.3,
                  476.7, 478.7, 479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486.7]
    elif T_in == 330:
        T_list = [330, 357.3, 382, 406.6, 431.3, 455.1, 461.5, 467.8, 473.1,
                  476.8, 478.8, 479.4, 480.1, 481.4, 482.4, 483.4, 484.4, 485.4, 486.7]
    elif T_in == 340:
        T_list = [340, 365.3, 390, 414.6, 439.3, 458.6, 464, 469.3, 474.6,
                  478, 479, 479.6, 480.3, 481.6, 482.6, 483.6, 484.6, 485.6, 486.7]
    elif T_in == 350:
        T_list = [350, 374.7, 399.3, 424, 449.3, 460.7, 466, 471.3, 476.7,
                  478.7, 479.3, 480, 481.3, 482.3, 483.3, 484.3, 485.3, 486, 486.7]

    n_steps = 1000
    n_pfr = 18
    length = 18
    area = 3.14 * (202.3 / 1000) ** 2 / 4

    compositions, t = EDC_coking(
        reaction_mech_x,
        T_list,
        pressure_0,
        CCl4_X_0,
        CHCl3_X_0,
        Tri_X_0,
        CP_X_0,
        mass_flow_rate,
        n_steps,
        n_pfr,
        length,
        area
    )
    information = ''
    for j in range(n_pfr):
        information += str(T_list[j]) + ',' + str(T_list[j+1]) + ','
        for k in compositions[j]:
            information += str(k) + ','
        information += str(pressure_0) + ',' + str(CCl4_X_0) + ',' + str(CHCl3_X_0) \
            + ',' + str(Tri_X_0) + ',' + str(CP_X_0) + ',' + \
            str(sum(t[:j+1])) + ',' + str(t[j]) + '\n'
        # information += str(pressure_0) + ',' + str(CCl4_X_0) + ',' + str(sum(t[:j+1])) + ',' + str(t[j]) + ','
        # information += str(cracking_rates[j+1]) + '\n'

    f.write(information)

    f.close()
    print('Generating training data finished')
    print('Training data creation terminated at {0}'.format(time.asctime()))


if __name__ == '__main__':
    main()
