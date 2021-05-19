
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt


CCl4_X_0 = float(1000)
pressure_0 = 12.4
reaction_mech="~/EDC/FPC/KM/2009_Schirmeister_EDC/chem_annotated_irreversible.cti"
mass_flow_rate = float(36)
length = 18
n_steps = 1000
area = 0.03225097679
n_pfr=18
# T_list = [322, 350, 375, 399, 424, 451, 461, 466, 471, 477, 479, 479, 480, 481, 482, 483, 484, 485, 486]

temp_T = 750
if CCl4_X_0 > 1 :  #ppm
        CCl4_X_0 = CCl4_X_0 /1000000
# T_0 = 273.15 + T_list[0]  # inlet temperature [K]
pressure_0 *= ct.one_atm
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
gas = ct.Solution(reaction_mech)
gas.TPX = temp_T, pressure_0, composition_0
dz = length / n_steps
r_vol = area * dz

# create a new reactor
r = ct.IdealGasConstPressureReactor(gas)
r.volume = r_vol

gas.set_multiplier(10)

# # create a reservoir to represent the reactor immediately upstream. Note
# # that the gas object is set already to the state of the upstream reactor
# upstream = ct.Reservoir(gas, name='upstream')

# # create a reservoir for the reactor to exhaust into. The composition of
# # this reservoir is irrelevant.
# downstream = ct.Reservoir(gas, name='downstream')

# # The mass flow rate into the reactor will be fixed by using a
# # MassFlowController object.
# m = ct.MassFlowController(upstream, r, mdot=mass_flow_rate)

# # We need an outlet to the downstream reservoir. This will determine the
# # pressure in the reactor. The value of K will only affect the transient
# # pressure difference.
# v = ct.PressureController(r, downstream, master=m, K=1e-5)

sim = ct.ReactorNet([r])

rxn = gas.reaction_equations()
n_rxn = len(rxn)

for i in range(n_rxn):
    r.add_sensitivity_reaction(i)

max_s = [0] * n_rxn
max_s_abs = [0]* n_rxn
sim.rtol = 1e-6 # Relative tolerance
sim.atol = 1e-15 # Absolute tolerance

sim.rtol_sensitivity = 1e-6 # Relative tolerance for sensitivity
sim.atol_sensitivity - 1e-6 # absolute tolerance for sensitivity

# define time, space, and other information vectors  
z = (np.arange(n_steps) + 1) * dz
t = np.zeros(n_pfr)  # residence time in each PFR reactor
compositions = [None] * n_pfr  # compositions of output stream in each PFR reactor
states = ct.SolutionArray(gas,extra = ['t'])

tot_time = 2e-3
dt = 5e-6

print(r.component_index('VCM(10)'))
for time in np.arange(0,tot_time,dt):
    sim.advance(time)
    states.append(r.thermo.state,t = time*1000)
    

    for j in range(n_rxn):
        s = sim.sensitivity('EDC(1)',j)
        

        if np.abs(s) > np.abs(max_s[j]):
            max_s[j] = s
            max_s_abs[j] = abs(s)
sen_zip = zip(max_s_abs,max_s,range(n_rxn))
sen_tuple = tuple(sen_zip)
sen_tuple_sorted = sorted(sen_tuple,reverse=True)

top_n = 10
top_rxn_index = [0]*top_n
top_rxn = [0]*top_n
top_sen = [0]*top_n

for i in range(top_n):
    top_rxn_index[i] = int(sen_tuple_sorted[i][2])
    top_rxn[i] = rxn[top_rxn_index[i]]
    top_sen[i] = float(sen_tuple_sorted[i][1])

print("The top %d EDC sensitive reactions are:" %(top_n))
print("-----------------------------------------------------------")
print("%10s  %s %18s %12s %12s" % ('Reaction Index','|','Reaction','|','Sensitivity'))
print("-----------------------------------------------------------")
for i in range(top_n):
    print("%7d %8s %26s %4s %10.8f" % (top_rxn_index[i],'|',top_rxn[i],'|',top_sen[i]))
print("-----------------------------------------------------------")

top_rxn_index.reverse()
top_rxn.reverse()
top_sen.reverse()

plt.figure(1)
plt.barh(range(top_n),top_sen,0.75,align= 'center')
plt.yticks(range(top_n),top_rxn)
plt.xlabel('')
plt.title('{0} Most EDC Sensitivity Reactions' .format(top_n), fontweight= 'bold')
plt.grid(which='major')
plt.minorticks_on()
plt.grid(which='minor',alpha=0.2)
plt.tight_layout()
plt.savefig('sensitivity.png')








# for i, T in enumerate(T_list[1:]):
#     Ti = T_list[i] + 273.15
#     Te = T + 273.15
#     dT = (Te - Ti) / n_steps
#     T = Ti
#     t_r = np.zeros_like(z)  # residence time in each CSTR reactor
#     # iterate through the PFR cells
#     for n in range(n_steps):
#         # simulate the linear T-profile in each reactor
#         T = Ti + (n + 1) * dT
#         gas.TP = T, None
#         r.syncState()
#         # Set the state of the reservoir to match that of the previous reactor
#         gas.TPX = r.thermo.TPX
#         upstream.syncState()
#         # integrate the reactor forward in time until steady state is reached
#         sim.reinitialize()
#         sim.set_initial_time(0)
#         sim.advance_to_steady_state()
#         # compute velocity and transform into time
#         t_r[n] = r.mass / mass_flow_rate  # residence time in this reactor
#         # write output data
#         states.append(r.thermo.state)