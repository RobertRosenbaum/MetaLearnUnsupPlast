import torch
import torch.nn as nn
import numpy as np

# GeneticMethodA is a genetic method that takes a population,
# a list of meta-loss values, a temperature, and a number of
# parents as input. The function does not return any output,
# but it modifies the meta-parameters of the population to
# be equal to the average parents' meta-parameters plus noise
# where noise is scaled by temperature.
# The parents are the agents in the population with the lowest meta-loss
# def GeneticMethodA(Pop,MetaLoss,Temperature,NumParents):
#
#   # Number of metaparams per agent
#   numMP=len(Pop[0].MetaParams)
#
#   # Sort the population
#   temp=torch.argsort(MetaLoss)
#   SortedPopulation=[Pop[ii] for ii in temp]
#
#   # Get all meta-params of sorted pop
#   AllMetaParams=torch.tesnor([p.MetaParams for p in SortedPopulation])
#
#   # Take the mean of the parents' meta-params
#   ParentMetaParams=torch.mean(AllMetaParams[:NumParents,:],axis=0)
#
#   # Set population meta-params
#   for j in range(len(Pop)):
#     Population[j].MetaParams=ParentMetaParams+Temperature*torch.randn(numMP)


# GeneticMethodB is a genetic method that returns a weighted
# average of the metaparams where weights are given by the
# deviation of the metalosses from a reference metaloss.
# It also subtracts a multiple of the sign of the meta-params,
# effectively imposing an L1 penalty
def GeneticMethodB(Pop, MetaLoss, Temperature, NumParents, MinWeight=.5, lam=0):
    # Number of metaparams per agent
    numMP = len(Pop[0].MetaParams)

    # Sort the population
    temp = torch.argsort(MetaLoss)
    SortedPopulation = [Pop[ii] for ii in temp]

    # Get all meta-params of sorted pop

    AllMetaParams = torch.stack([p.MetaParams for p in SortedPopulation])

    # weights for weighted average
    wts = torch.linspace(1, MinWeight, NumParents)

    # Take the weighted average of the parents' meta-params
    ParentMetaParams = torch.mean(wts[:,None]*AllMetaParams[:NumParents, :], axis=0)/wts.sum()

    # Set population meta-params
    for j in range(len(Pop)):
        Pop[j].MetaParams = ParentMetaParams + Temperature * torch.randn(numMP)- lam * torch.sign(
            ParentMetaParams)

