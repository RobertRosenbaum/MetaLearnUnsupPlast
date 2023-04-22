from Agents import FCFFwdUnsupAgent
from SupModules import FCFFwdSupModule
from MetaOptimizationMethods import GeneticMethodB
import torch
from SemiSupMethods import SemiSupMethod1
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Number of unsupervised and supervised
# iterations for semi-sup  learning
NumItUnsup = 10
NumItSup = 5#0

# Batch sizes
unsup_batch_size = 128
train_batch_size = 128
test_batch_size = 128

# Number of generations
NumGenerations=25

# Population size
PopSize=10#0

# Standard deviation for initializing
# meta-params
InitialMetaParamStd = .001

# Evolutionary hyperparams
Temperature=InitialMetaParamStd
NumParents=10#(int)(PopSize/2)
lam = 0


# Hyperparams for unsup agents
UnsupHyperParams = {}
UnsupMetaParams = .1 * torch.rand(8)
UnsupHyperParams["Depth"]=5
UnsupHyperParams["InWidth"]= 28 * 28
UnsupHyperParams["HiddenWidth"]=100
UnsupHyperParams["OutWidth"]=100
UnsupHyperParams["gHidden"]=torch.tanh#nn.ReLU()
UnsupHyperParams["gLast"]=torch.tanh#nn.ReLU()

# Hyperparams for sup module
SupHyperParams = {}
SupHyperParams["Depth"]=3
SupHyperParams["InWidth"]= UnsupHyperParams["OutWidth"]
SupHyperParams["HiddenWidth"]=50
SupHyperParams["OutWidth"]=10
SupHyperParams["gHidden"]=nn.ReLU()
SupHyperParams["gLast"]=nn.Identity()

# Function to get an initialized supervised learning module
def SupModuleGetter():
    return FCFFwdSupModule(SupHyperParams)

# Import and build data loaders
from torchvision.datasets import MNIST
train_dataset = MNIST('./',
      train=True,
      transform=transforms.ToTensor(),
      download=True)
test_dataset = MNIST('./',
      train=False,
      transform=transforms.ToTensor(),
      download=True)
unsup_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=unsup_batch_size,
                                          shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=test_batch_size,
                                          shuffle=True)


# iInitialize meta-parameters
NumMetaParams = 8
MetaParams=torch.zeros(NumGenerations,PopSize,NumMetaParams)
MetaParams[0,:,:]= InitialMetaParamStd * torch.randn(PopSize, NumMetaParams)

# Supervised loss function and optimizer
SupLossFun = nn.CrossEntropyLoss()
SupOptimizerGetter = torch.optim.Adam

# Semi-supervised learning method
method1 = SemiSupMethod1(SupModuleGetter, SupLossFun, SupOptimizerGetter, NumItUnsup, NumItSup)

# Create a population of unsupervised learning agents
Population=[FCFFwdUnsupAgent(MetaParams[0,j,:], UnsupHyperParams) for j in range(PopSize)]

method1.UnsupLifetime(Population, unsup_loader)
MetaLoss = method1.ComputeMetaLoss(Population, train_loader, test_loader)
print(MetaLoss.mean())

plt.figure()
plt.hist(MetaLoss)
plt.show()

GeneticMethodB(Population, MetaLoss, Temperature, NumParents, .5, lam)


print('done')

