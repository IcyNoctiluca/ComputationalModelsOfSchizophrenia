#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: s1867389
"""


import numpy as np
from matplotlib import pyplot as plt


# time scale of recovery variable u
a = 0.02

# sensitivity of recovery variable u
b = 0.25

# reset potential of neuron after spike
c = -65                     # mV

# after spike reset of recovery variable u
d = 2

# auxiliary after spike reseting
vMax = 30                   # mV

# injected current
I_e = 10

# length of time step (ms)
dt = 0.1

# simulation length 500ms
simLength = int(500 / dt)


# returns the rate of change of voltage as given by Izhikevich (1)
def voltageChangeRate(voltage, current, u):
    return 0.04 * voltage ** 2 + 5 * voltage + 140 - u + current


# returns the rate of change of membrane recovery variable u as given by Izhikevich (2)
def membraneRecoveryChangeRate(a, b, voltage, u):
    return a * (b * voltage - u)


# in time steps of 0.1ms
timesteps = np.arange(0,simLength)

# current over simulation. 0 for first 20% of simulation length, 2 for remainder
I = np.append(np.zeros(int(0.2 * simLength)), I_e * np.ones(int(0.8 * simLength)))

# record membrane potential V for the smiulation period
vSpiking = np.zeros(simLength)
membraneRecovery = np.zeros(simLength)
membraneRecovery[0] = 10
vSpiking[0] = c                         # initial value is at resting potential

vPlot = np.copy(vSpiking)               # make copy of recorded voltage values


# compute in each timestep of simulation
for t in range(simLength - 1):

    vSpiking[t + 1] = vSpiking[t] + voltageChangeRate(vSpiking[t], I[t], membraneRecovery[t]) * dt
    membraneRecovery[t + 1] = membraneRecovery[t] + membraneRecoveryChangeRate(a, b, vSpiking[t], membraneRecovery[t]) * dt

    vPlot[t + 1] = vSpiking[t + 1]

    if vSpiking[t + 1] >= vMax:
        vSpiking[t + 1] = c
        membraneRecovery[t + 1] += d


plt.plot(timesteps[500:], vPlot[500:])


plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")


plt.show()
