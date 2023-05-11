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
    def __init__(self, SupModuleGetter, SupLossFun, SupOptimizerGetter, NumItUnsup, NumItSup, device = 'cpu'):

        # Module for creating supervised agents
        # Calling:
        # model = SupModuleGetter()
        # should return a model for supervised learning
        self.SupModuleGetter = SupModuleGetter

        # An optimizer for supervised learning
        # e.g., SupOptimizerGetter = torch.optim.Adam
        self.SupOptimizerGetter = SupOptimizerGetter

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

        # Set the device on which to run method.
        # All parameters of the population must be
        # on this device.
        self.device = device

    # Function to simulate one generation of a population of agents.
    # This corresponds to iterating the unsupervised learning rule.
    def UnsupLifetime(self, Population, unsup_loader):

        # Putting this outside of the loop across
        # populations means that data is not
        # drawn iid with replacement across agents,
        # but I think it's still okay
        UnsupIterator = iter(unsup_loader)


        for j in range(len(Population)):

            # Initialize unsupervised weights
            Population[j].InitializeParams()

            for k in range(self.NumItUnsup):
                try:
                    X,_=next(UnsupIterator)
                except:
                    UnsupIterator = iter(unsup_loader)
                    X, _ = next(UnsupIterator)

                # Put X on Agent Device
                X=X.to(Population[j].device)

                # Apply network to a batch of inputs
                V = Population[j](X)

                # Update params
                Population[j].UpdateParams()

    # Function to compute meta-loss using the current meta-params in population.
    # This corresponds to performing a supervised learning loop
    # and then computing error rate on a test data set.
    def ComputeMetaLoss(self, Population, train_loader, test_loader):
        self.TrainLossCurve = torch.zeros(self.NumItSup)
        self.TrainAccuracyCurve = torch.zeros(self.NumItSup)
        self.MeanTestLoss = 0

        # Initialize data iterators
        TrainIterator = iter(train_loader)
        TestIterator = iter(test_loader)

        # For each agent and each iteration
        Errors = torch.zeros(len(Population))
        for j in range(len(Population)):

            # Initialize new supervised model and
            # optimizer
            SupModel = self.SupModuleGetter().to(Population[j].device)
            optimizer = self.SupOptimizerGetter(SupModel.parameters())


            # Perform supervised learning
            for k in range(self.NumItSup):
                try:
                    Xtrain,Ytrain = next(TrainIterator)
                except:
                    TrainIterator = iter(train_loader)
                    Xtrain, Ytrain = next(TrainIterator)

                # Put data on Agent Device
                Xtrain = Xtrain.to(Population[j].device)
                Ytrain = Ytrain.to(Population[j].device)


                # Pre-process with agent
                V = Population[j](Xtrain)

                # Get output and loss
                Yhat = SupModel(V)
                Loss = self.SupLossFun(Yhat, Ytrain)

                # Update using optimizer passed in
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

                # Store pop avg train accuracy and loss
                with torch.no_grad():
                    Guesses = torch.argmax(Yhat, axis=1)
                    self.TrainAccuracyCurve[k] += torch.mean((Guesses == Ytrain).float().cpu()) / len(Population)
                    self.TrainLossCurve[k] += Loss.item()/len(Population)


            with torch.no_grad():

                # Get test data
                try:
                    Xtest,Ytest = next(TestIterator)
                except:
                    TestIterator = iter(test_loader)
                    Xtest, Ytest = next(TestIterator)

                # Put data on Agent Device
                Xtest = Xtest.to(Population[j].device)
                Ytest = Ytest.to(Population[j].device)

                # Pre-process with agent
                V = Population[j](Xtest)

                # Get output and loss
                Yhat = SupModel(V)
                self.MeanTestLoss += self.SupLossFun(Yhat, Ytest)/len(Population)

                # Compute error rate
                Guesses = torch.argmax(Yhat, axis=1)
                Accuracy = torch.mean((Guesses == Ytest).float().cpu())
                Errors[j] = 1-Accuracy
        return Errors


