# -*- coding: utf-8 -*-
# from EDC.FPC.ML.ML_model_V2 import CCl4_X_0
import time
import random
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import sys
import os



def EDC_cracking(
        reaction_mech,
        T_list,
        pressure_0,
        CCl4_X_0,
        mass_flow_rate,
        n_steps=1000,
        n_pfr=18,
        length=18,
        area=0.02724553665,
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
    if CCl4_X_0 > 1 :  #ppm
        CCl4_X_0 = float(CCl4_X_0) / 1000000
    spcs = ct.Species.listFromFile(reaction_mech)
    for spc in spcs[::-1]:
        if spc.composition == {'C': 2.0, 'Cl': 2.0, 'H': 4.0} and spc.charge == 0:
            EDC_label = spc.name        
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
        cracking_rates.append(cracking_rate)
    return compositions, t, cracking_rates
    
# def EDC_cracking_C2H3Cl3(
#         reaction_mech,
#         T_list,
#         pressure_0,
#         CCl4_X_0,
#         mass_flow_rate,
#         n_steps=1000,
#         n_pfr=18,
#         length=18,
#         area=0.027403710696000005,
#         label=None,
#         ):
#     """
#     Module that runs a single PFR Cantera simulation via a series of CSTRs.
#     The Plug flow reactor is represented by a linear chain of zero-dimensional
#     reactors. The gas at the inlet to the first one has the specified inlet
#     composition, and for all others the inlet composition is fixed at the
#     composition of the reactor immediately upstream. Since in a PFR model there
#     is no diffusion, the upstream reactors are not affected by any downstream
#     reactors, and therefore the problem may be solved by simply marching from
#     the first to last reactor, integrating each one to steady state.
#     Parameters
#     =============== =============================================================
#     Attribute       Description
#     =============== =============================================================
#     `reaction_mech` Cantera reaction mechanism (.cti file)
#     `T_list`        Temperature profile (°C)
#     `pressure_0`    Initial pressue (atm)
#     `CCl4_X_0`      Initial CCl4 concentration (mass fraction)
#     `mass_flow_rate`Mass flow rate of input gas (T/H)      
#     `n_steps`       Number of iterations/number of CSTRs
#     `n_pfr`         Number of PFRs
#     `length`        Length of each PFR (m)
#     `area`          Cross-sectional area (m**2)
#     `label`         Label of this mechanism
#     =============== =============================================================
    
    
#     """
#     #######################################################################
#     # Input Parameters
#     #######################################################################
#     T_0 = 273.15 + T_list[0]  # inlet temperature [K]
#     pressure_0 *= ct.one_atm
#     spcs = ct.Species.listFromFile(reaction_mech)
#     for spc in spcs[::-1]:
#         if spc.composition == {'C': 2.0, 'Cl': 2.0, 'H': 4.0} and spc.charge == 0:
#             EDC_label = spc.name   
#         if spc.composition == {'C': 2.0, 'Cl': 3.0, 'H': 3.0} and spc.charge == 0:
#             C2H3Cl3_label = spc.name
#         if spc.composition == {'C': 1.0, 'Cl': 4.0} and spc.charge == 0:
#             CCl4_label = spc.name
#     EDC_X_0 = 1 - CCl4_X_0
#     composition_0 = '{}:{}, {}:{}'.format(EDC_label, EDC_X_0, CCl4_label, CCl4_X_0)
#     mass_flow_rate *= 1000 / 3600  # T/H to kg/s
    
#     # import the gas model and set the initial conditions
#     model = ct.Solution(reaction_mech)
#     model.TPX = T_0, pressure_0, composition_0
#     dz = length / n_steps
#     r_vol = area * dz
    
#     # create a new reactor
#     r = ct.IdealGasReactor(model)
#     r.volume = r_vol
    
#     # create a reservoir to represent the reactor immediately upstream. Note
#     # that the gas object is set already to the state of the upstream reactor
#     upstream = ct.Reservoir(model, name='upstream')
    
#     # create a reservoir for the reactor to exhaust into. The composition of
#     # this reservoir is irrelevant.
#     downstream = ct.Reservoir(model, name='downstream')
    
#     # The mass flow rate into the reactor will be fixed by using a
#     # MassFlowController object.
#     m = ct.MassFlowController(upstream, r, mdot=mass_flow_rate)
    
#     # We need an outlet to the downstream reservoir. This will determine the
#     # pressure in the reactor. The value of K will only affect the transient
#     # pressure difference.
#     v = ct.PressureController(r, downstream, master=m, K=1e-5)
    
#     sim = ct.ReactorNet([r])
    
#     # define time, space, and other information vectors  
#     z = (np.arange(n_steps) + 1) * dz
#     t = np.zeros(n_pfr)  # residence time in each PFR reactor
#     compositions = [None] * n_pfr  # compositions of output stream in each PFR reactor
#     states = ct.SolutionArray(r.thermo)
    
#     cracking_rates = [0]
#     forming_rates = [0]
#     for i, T in enumerate(T_list[1:]):
#         Ti = T_list[i] + 273.15
#         Te = T + 273.15
#         dT = (Te - Ti) / n_steps
#         T = Ti
#         t_r = np.zeros_like(z)  # residence time in each CSTR reactor
#         # iterate through the PFR cells
#         for n in range(n_steps):
#             # simulate the linear T-profile in each reactor
#             T = Ti + (n + 1) * dT
#             model.TP = T, None
#             r.syncState()
#             # Set the state of the reservoir to match that of the previous reactor
#             model.TPX = r.thermo.TPX
#             upstream.syncState()
#             # integrate the reactor forward in time until steady state is reached
#             sim.reinitialize()
#             sim.set_initial_time(0)
#             sim.advance_to_steady_state()
#             # compute velocity and transform into time
#             t_r[n] = r.mass / mass_flow_rate  # residence time in this reactor
#             # write output data
#             states.append(r.thermo.state)
#         t[i] = np.sum(t_r)
#         compositions[i] = model.X[4:]
#         cracking_rate = (EDC_X_0 - model.X[model.species_index(EDC_label)]) / EDC_X_0
#         forming_rate = model.X[model.species_index(C2H3Cl3_label)]
#         forming_rates.append(forming_rate)
#         cracking_rates.append(cracking_rate)
#     return compositions, t, cracking_rates ,forming_rates

    
def plot(T_list, cracking_rates):
    
    ndata = len(T_list)
    fig, ax1 = plt.subplots()
    
    l1 = ax1.plot(range(ndata), T_list, color='r', marker='o', label='Temperature ($^\circ$C)')
    ax1.set_ylabel('Temperature ($^\circ$C)')
    ax1.set_ylim(0, 600)
    
    ax2 = ax1.twinx()
    cracking_rates = [i * 100 for i in cracking_rates]
    l2 = ax2.plot(range(ndata), cracking_rates, color='b', marker='o', label='Cracking rates (%)')
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
    T_list = [322, 350, 375, 399, 424, 451, 461, 466, 471, 477, 479, 479, 480, 481, 482, 483, 484, 485, 486]
    pressure_0 = 11.4
    CCl4_X_0 = 0.001
    mass_flow_rate = 72
    n_steps = 100
    n_pfr=18
    length = 18
    area = 3.14 * ((8 * 2.54 - 2 * 0.818)/ 100) ** 2 / 4 
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
   
    
    # reaction_mech_x = '../KM//2009_Schirmeister_EDC/chem_annotated_irreversible.cti'
    reaction_mech_x = '../KM//2009_Schirmeister_EDC/test.cti'
    # reaction_mech_y = '../KM/2001_Choi_EDC/chem_annotated_irreversible.cti'
    
    # Cantera simulation parameters
    n_steps = 1000
    n_pfr = 18
    length = 18
    # area = 3.14 * ((8 * 2.54 - 2 * 0.818)/ 100) ** 2 / 4
    
    n = 1
    mass_flow_rate = float(sys.argv[1])
    pressure_0 = float(sys.argv[2])
    T_in = float(sys.argv[3])
    # T_out = float(sys.argv[3])
    CCl4_X_0 = float(sys.argv[4])

    # mass_flow_rate = 36
    # pressure_0 = 11.4
    # T_in = 320
    # CCl4_X_0 = 1000
    
    write_header = True
    #21 CCl4
    # for CCl4_X_0 in np.linspace(End - 0.0001 , End , 2,endpoint=False):
    # for mass_flow_rate in [23, 27, 32, 36]:
    #     for pressure_0 in np.linspace(10.4, 14.4, 6):
    #         for T_in in np.linspace(300,350,6):   
    print('Training data creation initiated at {0}'.format(time.asctime()))
    if mass_flow_rate < 40:
        DATADIR = 'Data/FPCV8_newparams'
        # DATADIR='Data/FPC_cracking_V4_3m_irrev'
    else:
        DATADIR='Data/training_data_cracking_V8_3m_choi_rev_big'
    DATANAME=f'{mass_flow_rate}_{T_in}_{CCl4_X_0}_{pressure_0}.csv'
    # if not os.path.exists(DATADIR):
    #     os.makedirs(DATADIR)
    # if os.path.exists(f'{DATADIR}/{mass_flow_rate}_{pressure_0}_{T_in}_{CCl4_X_0}.csv'):
    #     sys.exit()
    # f = open(f'{DATADIR}/{mass_flow_rate}_{pressure_0}_{T_in}_{CCl4_X_0}.csv' , 'a')
    # if os.stat(f'{DATADIR}/{mass_flow_rate}_{pressure_0}_{T_in}_{CCl4_X_0}.csv').st_size ==0:
    #     f.write('Ti,Te,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,pressure_0,CCl4_X_0,t,tr,X\n')

    if not os.path.exists(DATADIR):
        os.makedirs(DATADIR)
    if os.path.exists(f'{DATADIR}/{DATANAME}'):
        sys.exit()
    f = open(f'{DATADIR}/{DATANAME}' , 'a')
    if os.stat(f'{DATADIR}/{DATANAME}').st_size ==0:{
        f.write('mass,Ti,Te,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,pressure_0,CCl4_X_0,t,tr,prev_X,X\n')
        # f.write('Ti,Te,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,pressure_0,CCl4_X_0,t,tr,X_sh,X_choi\n')

    }
    
    # t_0 = Tin t_5 = 453 t_9 = 478  t_19 = 486
    # T_list = [T_in , 453 ,478 ,486]
    # t 1 ~ 4
    

    # print('{}/10000 Cantera simulation...'.format(n))
    # success = False
    # while not success:
    #     while True:
    #         T_list = [T_in, 453 ,486]
    #         for i in range(n_pfr-2):
    #             T_list.append(random.uniform(T_in, 486))
    #         T_list.sort()
    #         if all(T_list >= np.linspace(T_in, 486, n_pfr+1)):
    #             break

    # T_list = [322, 350, 375, 399, 424, 451, 461, 466, 471, 477, 479, 479, 480, 481, 482, 483, 484, 485, 486]
    # pressure_0 = 11.4
    # CCl4_X_0 = End
    # mass_flow_rate = 72
    T_list = []
    if T_in == 300:
        T_list = [300,332,361,387,411,429.6,443.4,453.7,460.5,464.9,468.7,473,477.8,480.5,481.5,482.5,483.5,484.5,486.7]
    elif T_in == 310:
        T_list = [310,340,368.4,393.7,416.5,441.6,452,458.3,464.6,471,474,476.5,478.8,481.1,482.1,483.1,484.1,485.1,486.7]
    elif T_in == 320:
        T_list = [320,348,374.7,399.3,424,451.3,460.7,466,471.3,476.7,478.7,479.3,480,481.3,482.3,483.3,484.3,485.3,486.7]
    elif T_in == 330:
        T_list = [330,357.3,382,406.6,431.3,455.1,461.5,467.8,473.1,476.8,478.8,479.4,480.1,481.4,482.4,483.4,484.4,485.4,486.7]
    elif T_in == 340:
        T_list =[340, 365.3, 390, 414.6, 439.3, 458.6, 464, 469.3, 474.6, 478,
             479, 479.6, 480.3, 481.6, 482.6, 483.6, 484.6, 485.6, 486.7]
    elif T_in == 350:
        T_list =[350,374.7,399.3,424,449.3,460.7,466,471.3,476.7,478.7,479.3,480,481.3,482.3,483.3,484.3,485.3,486,486.7]
    # T_list = [320,348,374.7,399.3,424,451.3,460.7,462,466,468,471.3,474.3,476.7,477.7,478.7,479.3,480,481.3,482.3,483.3,484.3,485.3,486.7]
    n_steps = 1000
    n_pfr = 18
    length = 18
    area = 3.14 * (202.3/ 1000) ** 2 / 4 
    # area = 3.14 * ((262)/ 1000) ** 2 / 4  
    

    # compositions, t, _ = EDC_cracking(
    #                             reaction_mech_x,
    #                             T_list,
    #                             pressure_0,
    #                             CCl4_X_0,
    #                             mass_flow_rate,
    #                             n_steps,
    #                             n_pfr,
    #                             length,
    #                             area,
    #                             )
    compositions, t, cracking_sh = EDC_cracking(
                            reaction_mech_x,
                            T_list,
                            pressure_0,
                            CCl4_X_0,
                            mass_flow_rate,
                            n_steps,
                            n_pfr,
                            length,
                            area,
                            )                            
    
    # _, _, cracking_rates = EDC_cracking(
    #                             reaction_mech_y,
    #                             T_list,
    #                             pressure_0,
    #                             CCl4_X_0,
    #                             mass_flow_rate,
    #                             n_steps,
    #                             n_pfr,
    #                             length,
    #                             area,
    #                             )
    
    # _, _, cracking_rates, forming_rates = EDC_cracking_C2H3Cl3(
    #                             reaction_mech_y,
    #                             T_list,
    #                             pressure_0,
    #                             CCl4_X_0,
    #                             mass_flow_rate,
    #                             n_steps,
    #                             n_pfr,
    #                             length,
    #                             area,
    #                             )
    
    information = ''
    for j in range(n_pfr):
        information += str(mass_flow_rate) + ',' + str(T_list[j]) + ',' + str(T_list[j+1]) + ','
        for k in compositions[j]:
            information += str(k) + ','
        information += str(pressure_0) + ',' + str(CCl4_X_0) + ',' + str(sum(t[:j+1])) + ',' + str(t[j]) + ','+str(cracking_sh[j])+',' +str(cracking_sh[j+1]) + '\n'
        # information += str(cracking_sh[j+1]) + '\n'
        
    f.write(information)
    success = True
            # n += 1



    f.close()
    print('Generating training data finished')
    print('Training data creation terminated at {0}'.format(time.asctime()))

if __name__ == '__main__':
    main()