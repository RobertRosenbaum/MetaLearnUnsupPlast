from Agents import FCFwdNet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

MetaParams = .1*torch.rand(8)
HyperParams = {}

def gHidden0(x):
    return x

def gLast0(x):
    return x

HyperParams["Depth"]=5
HyperParams["InWidth"]=28*28
HyperParams["HiddenWidth"]=1000
HyperParams["OutWidth"]=10
HyperParams["gHidden"]=torch.tanh#nn.ReLU()
HyperParams["gLast"]=torch.tanh#nn.ReLU()

model = FCFwdNet(MetaParams, HyperParams)

print(model)


X=torch.randn(128, model.InWidth)
V=model(X)

for i in range(model.Depth+1):
    print('a',i,model.Activations[i].var().item())

model.UpdateParams()
V=model(X)
model.UpdateParams()
V=model(X)
model.UpdateParams()
# #print('!',model.Activations[2].shape)
# L=torch.norm(V)
# print(L)
#
#
#
# #wtemp=model.LinLayers[1].weight.data
#
#
# # TEST TO SEE IF WEIGHTS REALLY CHANGE
#
# L.backward()
# #print(model.LinLayers[0].weight.grad.shape)


print('done')

