# This file defines semi-supervised learning methods.
# A Semi Supervised method should have the following functions:
# UnsupLifetime(Population, X)
# which simulated one lifetime (ie, one generation) of the population
# This will correspond to iterating the unsup learning rule
#
# ComputeMetaLoss(Population, X, Y, Xtest, Ytest)
# which computes the meta-loss. This will correspond to training the
# supervised learner on input,label batch X,Y and then computing
# test loss on input,label batch Xtest,Ytest
#
# Both functions should not return anything, but they modify Population
# which should be a Python list of agents.

import torch
import torch.nn as nn
import numpy as np

# SemiSupOneLayerMSE is a Semi Supervised method that has one output layer
# and uses the MSE (L2) loss function.

class SemiSupOneLayerMSE(object):
    def __init__(self, NumItUnsup, NumItSup, lr, mUnsup, mSup, mTestSup):
        # Number of iterations for unsupervised learning
        self.NumItUnsup = NumItUnsup

        # Number of iterations for supervised learning
        self.NumItSup = NumItSup

        # Learning rate for supervised learning
        self.lr = lr

        # Loss and accuracy curves
        self.TrainLossCurve = np.zeros(self.NumItSup)
        self.TrainAccuracyCurve = np.zeros(self.NumItSup)

    # Function to simulate one generation of a population of agents.
    # This corresponds to iterating the unsupervised learning rule.
    def UnsupLifetime(self, Population, X):
        # Number of data points
        NumData = X.shape[1]

        # Initialize data counter
        iSampleUnsup = 0

        # For each agent and each iteration
        for j in range(len(Population)):
            for k in range(self.NumItUnsup):

                # Sample data for unsup learning
                if (iSampleUnsup + self.mUnsup) >= NumData:
                    iSampleUnsup = 0
                    Ishuffle = np.random.permutation(NumData)
                    X = X[:, Ishuffle]
                Isample = range(iSampleUnsup, iSampleUnsup + self.mUnsup)
                Xsample = X[:, Isample]
                iSampleUnsup = iSampleUnsup + self.mUnsup

                # Apply network to a batch of inputs
                V = Population[j].Apply(Xsample)

                # Update params
                Population[j].UpdateParams()


