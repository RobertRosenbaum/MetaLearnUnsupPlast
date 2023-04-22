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

class SemiSupMethod1(object):
    def __init__(self, SupModule, SupLossFun, SupOptimizer, NumItUnsup, NumItSup):

        # Module for creating supervised agents
        # Calling:
        # model = SupModule()
        # should return an appropriate model for supervised learning
        self.SupModule = SupModule

        # An optimizer for supervised learning
        # e.g., SupOptimizer = torch.optim.Adam
        self.SupOptimizer = SupOptimizer

        # A loss function for supervised learning
        # e.g., SupLossFun = nn.CrossEntropyLoss()
        self.SupLossFun = SupLossFun

        # Number of iterations for unsupervised learning
        self.NumItUnsup = NumItUnsup

        # Number of iterations for supervised learning
        self.NumItSup = NumItSup

        # Loss and accuracy curves
        self.TrainLossCurve = torch.zeros(self.NumItSup)
        self.TrainAccuracyCurve = torch.zeros(self.NumItSup)
        self.MeanTestLoss = -1

    # Function to simulate one generation of a population of agents.
    # This corresponds to iterating the unsupervised learning rule.
    def UnsupLifetime(self, Population, unsup_loader):
        for j in range(len(Population)):
            UnsupIterator = iter(unsup_loader)
            for k in range(self.NumItUnsup):
                try:
                    X,_=next(UnsupIterator)
                except:
                    UnsupIterator = iter(unsup_loader)
                    X, _ = next(UnsupIterator)

                # Apply network to a batch of inputs
                V = Population[j].Apply(X)

                # Update params
                Population[j].UpdateParams()

    # Function to compute meta-loss using the current meta-params in population.
    # This corresponds to performing a supervised learning loop
    # and then computing error rate on a test data set.
    def ComputeMetaLoss(self, Population, train_loader, test_loader):
        self.TrainLossCurve = np.zeros(self.NumItSup)
        self.TrainAccuracyCurve = np.zeros(self.NumItSup)
        self.MeanTestLoss = 0


        TrainIterator = iter(train_loader)
        TestIterator = iter(test_loader)

        # For each agent and each iteration
        losses = torch.zeros(len(Population))
        ErrorRate = torch.zeros(len(Population))
        for j in range(len(Population)):

            # Initialize new supervised model and
            # optimizer
            SupModel = self.SupModule()
            optimizer = self.SupOptimizer(SupModel.parameters())


            # Perform supervised learning
            for k in range(self.NumItSup):
                try:
                    Xtrain,Ytrain = next(TrainIterator)
                except:
                    TrainIterator = iter(train_loader)
                    Xtrain, Ytrain = next(TrainIterator)

                # Pre-process with agent
                V = Population[j].Apply(Xtrain)

                # Get output and loss
                Yhat = SupModel(V)
                Loss = self.SupLossFun(Yhat, Ytrain)



                # Update using optimizer passed in
                optimizer.zero_grad()
                Loss.backward()
                self.SupOptimizer().step()

                # Store pop avg train accuracy and loss
                with torch.no_grad():
                    Guesses = np.argmax(Yhat, axis=1)
                    self.TrainAccuracyCurve[k] += np.mean((Guesses == Ytrain)) / len(Population)
                    self.TrainLossCurve[k] += Loss.item()/len(Population)


            with torch.no_grad():

                # Get test data
                try:
                    Xtest,Ytest = next(TestIterator)
                except:
                    TestIterator = iter(test_loader)
                    Xtest, Ytest = next(TestIterator)

                # Pre-process with agent
                V = Population[j].Apply(Xtest)

                # Get output and loss
                Yhat = SupModel(V)
                self.MeanTestLoss += self.SupLossFun(Yhat, Ytest)/len(Population)

                # Compute error rate
                Guesses = np.argmax(Yhat, axis=0)
                Accuracy = np.mean((Guesses == Ytest))
                ErrorRate[j] = 1-Accuracy
        return ErrorRate


