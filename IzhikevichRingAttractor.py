#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: s1867389
"""


import numpy as np
from matplotlib import pyplot as plt
import random


# time scale of recovery variable u
a = 0.02

# sensitivity of recovery variable u
b = 0.2

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

# ring attractor net population
nNeurons = 10

# number of memories for network to remember
nMemories = 1


# randomly returns either 1 or -1 with equal likelihood
def randomMemoryBit():
    return -1 if random.random() > 0.5 else 1

# returns the rate of change of voltage as given by Izhikevich (1)
def voltageChangeRate(voltage, current, u):
    return 0.04 * voltage ** 2 + 5 * voltage + 140 - u + current

# definition of sigmoid function
def sigmoidActivation(input):
    return 1 / (1 + np.exp(-1 * input))


# returns the rate of change of membrane recovery variable u as given by Izhikevich (2)
def membraneRecoveryChangeRate(a, b, voltage, u):
    return a * (b * voltage - u)


# encodes a memory as an angle which can be used as a stimulus for the neurons
def encodeMemory(memory):

    # convert memory to a binary string
    binaryStringMem = ''
    for bit in memory:
        if bit == -1:
            binaryStringMem += str(0)
        else:
            binaryStringMem += str(bit)

    # conv binary string to decimal
    mem = int(binaryStringMem, 2)

    # the total number of possible binary strings of given length
    totalStates = 2 ** len(binaryStringMem)

    # angle of state of memory (each memory has a unique angle)
    return 2 * np.pi * mem / totalStates


# returns corresponding how close the stimulus is to the neurons preferred tuning angle
def getThalamicInput(neuron, stimulus):

    # the angle at which the neuron is tuned to repsond most
    # for N neurons, each with a preferred angle evenly distributed
    tunedAngle = neuron * 2 * np.pi / nNeurons

    # return a measure of how close the stimulus is to the neuron's tuned angle
    return np.cos(tunedAngle - stimulus)


# weights between two neurons
# characterised by nearby excitation and longer inhibition
def getWeight(n1, n2):

    # no link to itself
    if n1 == n2:
        return 0

    # compute angle between two neurons
    # where neurons are distributed evenly around a ring
    angle = 2 * np.pi * np.abs(n1 - n2) * (1/nNeurons)

    # modulate angle to take the shorter route
    if angle > np.pi:
        angle = 2 * np.pi - angle


    # prune the connection if weight greater than a certain threshold
    if angle > 0.8 * np.pi:
        return 0

    return np.cos(angle)


# in time steps of 0.1ms
timesteps = np.arange(0,simLength)

# external input for each neuron
# ie the external cue encoded in a vector like a memory
memories = [[randomMemoryBit() for n in range(nNeurons)] for m in range(nMemories)]

# record membrane potentials V for the smiulation period for each neuron on the ring net
vSpiking = np.zeros(simLength * nNeurons).reshape(nNeurons, simLength)
membraneRecovery = np.zeros(simLength * nNeurons).reshape(nNeurons, simLength)
current = np.zeros(simLength * nNeurons).reshape(nNeurons, simLength)

membraneRecovery[:,0] = 10
vSpiking[:,0] = c                           # initial value is at resting potential
current[:,0] = 10

vPlot = np.copy(vSpiking)                   # make copy of recorded voltage values for plotting

# chose a memory to encode as a stimulus
stimulus = encodeMemory(memories[0])


# compute in each timestep of simulation
for t in range(simLength - 1):

    # compute potentials for each neuron
    for neuron in range(nNeurons):

        current[neuron, t + 1] = 5 * getThalamicInput(neuron, stimulus) * np.exp(-t * 1 / simLength) + 5 * np.arctan(np.sum([getWeight(neuron, n) * vPlot[n, t] for n in range(nNeurons)]))

        vSpiking[neuron, t + 1] = vSpiking[neuron, t] + voltageChangeRate(vSpiking[neuron, t], current[neuron, t + 1], membraneRecovery[neuron, t]) * dt

        membraneRecovery[neuron, t + 1] = membraneRecovery[neuron, t] + membraneRecoveryChangeRate(a, b, vSpiking[neuron, t], membraneRecovery[neuron, t]) * dt

        # value to plot before resetting the spike potential
        vPlot[neuron, t + 1] = vSpiking[neuron, t + 1]

        # reset the spike potential
        if vSpiking[neuron, t + 1] >= vMax:
            vSpiking[neuron, t + 1] = c
            membraneRecovery[neuron, t + 1] += d


# plot neuron activities

plt.plot(timesteps, vPlot[0, :])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Neuron 1")


plt.show()
plt.plot(timesteps, vPlot[1, :])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Neuron 2")


plt.show()
plt.plot(timesteps, vPlot[2, :])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Neuron 3")


plt.show()
plt.plot(timesteps, vPlot[3, :])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Neuron 4")


plt.show()
plt.plot(timesteps, vPlot[4, :])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Neuron 5")

plt.show()
plt.plot(timesteps, vPlot[5, :])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Neuron 6")


plt.show()
plt.plot(timesteps, vPlot[6, :])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Neuron 7")


plt.show()
plt.plot(timesteps, vPlot[7, :])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Neuron 8")


plt.show()
plt.plot(timesteps, vPlot[8, :])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Neuron 9")


plt.show()
plt.plot(timesteps, vPlot[9, :])
plt.title("Neuron 10")


plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")

plt.show()
