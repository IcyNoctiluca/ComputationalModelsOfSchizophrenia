#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: s1867389
"""

import numpy as np
import random
import scipy.spatial


# number of units for HF net
nNeurons = 100

# randomly generate some memories to store in the net
nMemories = 9

# hamming distance of input from target memory
corruptnessOfInput = 20

# scale factor T in the probability of neuron state update to 1
scaleFactor = 4

# coefficient of pruning of synapse
pruningCoef = 0.6


# randomly returns either 1 or -1 with equal likelihood
def randomMemoryBit():
    return -1 if random.random() > 0.5 else 1


# returns a memory of Hamming distance hammingDist away from the original memory mem
def degradeMemory(mem, hammingDist):

    # flips the state of a neuron 1 -> -1, and -1 -> 1
    flip = lambda a : -1 * a

    # initialise memory to be corrupted
    corruptedMemory = mem[:]

    # stores which neurons states for memory mem to flip ie [0,1,2,3,4] flips the states of the first 5 neurons
    locationsToFlip = set()

    # to ensure that duplicates neurons do not end up here
    # ensures there are exactly hammingDist unique neurons to flip
    while len(locationsToFlip) != hammingDist:
        locationsToFlip.add(random.randint(0, nNeurons - 1))

    # flip neuron state
    for loc in locationsToFlip:
        corruptedMemory[loc] = flip(mem[loc])

    return corruptedMemory


# returns the enery value of the network
def getNetEnergy(HFNetWeights, NFNetStates):

    return -0.5 * np.sum([HFNetWeights[n1][n2] * NFNetStates[n1] * NFNetStates[n2] for n1 in range(nNeurons) for n2 in range(nNeurons)])


# recovers the trained input from the HF net given a noisy version of the input
# asynchronous training
def convergeStatesToMininum(HFNetWeights, NFNetStates):

    # set initial energy state to worst possible value
    startEnergy = np.inf

    # if the current energy of the net is better than the current best
    while getNetEnergy(HFNetWeights, NFNetStates) < startEnergy:

        # update the current best
        startEnergy = getNetEnergy(HFNetWeights, NFNetStates)

        # for each neuron
        for n1 in range(nNeurons):

            # calculate the threshold at which the state is determined
            threshold = np.sum([NFNetStates[n2] * HFNetWeights[n1][n2] for n2 in range(nNeurons)])

            # likelihood that the a neuron is state is updated to 1
            probStateOn = 1 / (1 + np.exp( -1 * threshold / scaleFactor))

            # update the state
            if random.random() <= probStateOn:
                NFNetStates[n1] = 1
            else:
                NFNetStates[n1] = -1

    # return the converged net after no lower energy states found by further update
    return NFNetStates


# recovers the trained input from the HF net given a noisy version of the input
# synchronous training
def convergeStatesSynchronously(HFNetWeights, NFNetStates):

    # set initial energy state to worst possible value
    startEnergy = np.inf

    # if the current energy of the net is better than the current best
    while getNetEnergy(HFNetWeights, NFNetStates) < startEnergy:

        # update the current best
        startEnergy = getNetEnergy(HFNetWeights, NFNetStates)

        # create a copy of the net for synchronous training
        newHFNetStates = NFNetStates[:]

        # for each neuron
        for n1 in range(nNeurons):

            # calculate the threshold at which the state is determined
            # calculate based on the old net
            threshold = np.sum([NFNetStates[n2] * HFNetWeights[n1][n2] for n2 in range(nNeurons)])

            # likelihood that the a neuron is state is updated to 1
            probStateOn = 1 / (1 + np.exp( -1 * threshold / scaleFactor))

            # update the state
            if random.random() <= probStateOn:
                newHFNetStates[n1] = 1
            else:
                newHFNetStates[n1] = -1

        # reset the old network
        NFNetStates = newHFNetStates[:]

    # return the converged net after no lower energy states found by further update
    return newHFNetStates


# returns the hamming distance between two memories
# usually between the trained memory and the attractor state
def memoryHammingDist(mem1, mem2):
    return scipy.spatial.distance.hamming(mem1, mem2)


# returns the euclidean distance between two neurons
def euclideanNeuronalDistance(n1, n2):

    # row and column number of neuron
    # ie. if neuron number is 62, then its in the 6th index up the grid, 2nd index to the right
    # only works for 10 x 10 grid of neuronal sheet
    colN1 = np.where(np.arange(100).reshape(int(np.sqrt(100)), int(np.sqrt(100))) == n1)[0][0]
    rowN1 = np.where(np.arange(100).reshape(int(np.sqrt(100)), int(np.sqrt(100))) == n1)[1][0]

    colN2 = np.where(np.arange(100).reshape(int(np.sqrt(100)), int(np.sqrt(100))) == n2)[0][0]
    rowN2 = np.where(np.arange(100).reshape(int(np.sqrt(100)), int(np.sqrt(100))) == n2)[1][0]

    return np.sqrt((rowN2 - rowN1) ** 2 + (colN2 - colN1) ** 2)


# generates a list of nMemories memories, each memory is an nNeurons bits long list
memories = [[randomMemoryBit() for n in range(nNeurons)] for m in range(nMemories)]


# generate the HF network structure
HFNetWeights = np.zeros(nNeurons * nNeurons).reshape(nNeurons, nNeurons)
NFNetStates = np.zeros(nNeurons)


# setting the weights between pairs of neurons
for n1 in range(nNeurons):
    for n2 in range(nNeurons):

        # compute weight value based on sum of product of neuron states for each memory
        weight = np.sum([memory[n1] * memory[n2] for memory in memories])
        HFNetWeights[n1][n2] = weight

        # apply synaptic pruning by the rule
        if np.abs(weight) < pruningCoef * euclideanNeuronalDistance(n1, n2):
            HFNetWeights[n1][n2] = 0        # prune the connection

    HFNetWeights[n1][n1] = 0        # no link to itself


# to store errors in recovered memory from target
memoryErrors = []

# store differences between recovered memory with alternative memories
alternativeMemorySimilarities = []

# test network on recovering each memory
for mem in memories:

    # generate a corrupted memory
    corrMem = degradeMemory(mem, corruptnessOfInput)

    # sets the input state of neurons in the HF net to the corrupted memory
    for index, val in enumerate(corrMem):
        NFNetStates[index] = val


    # update the network based on corrupt input to recover the learned memory
    NFNetStates = convergeStatesToMininum(HFNetWeights, NFNetStates)

    # hamming distance between recovered and target memories
    memoryErrors.append(memoryHammingDist(NFNetStates, mem) * 100)

    # compare output with other memories tuned in the network
    alternativeMemorySimilarities.append(np.min([memoryHammingDist(NFNetStates, comparisonMem) * 100 for comparisonMem in memories if comparisonMem != mem]))


print (np.mean(memoryErrors), np.mean(alternativeMemorySimilarities))
