# This file defines different agents. Agents are nn modules with the following properties:
#   1) Agent.MetaParams should be a tensor of meta parameters.
#   2) Agent.forward should take a batch of inputs, X, and the output, V,
#      V is usually a batch of embedding vectors. The first dimension in X and V is batch size.
#   3) Agent.UpdateParams() should update the agent's params (e.g., apply the
#      plasticity rule). This update will typically depend on the internal state of
#      the agent (e.g., its activations) and also on the metaparams

import torch
import torch.nn as nn
import math
import numpy as np

###
# IdentityNet is a type of agent that just returns the input as the embedding
# It doesn't do anything. It is just for testing purposes.
###
class IdentityNet(nn.Module):
    def __init__(self, MetaParams):
        super(IdentityNet, self).__init__()
        self.MetaParams = MetaParams

    def forward(self, x):
        return x

    # Function for updating weights
    def UpdateParams(self):
        []


###
# FCFwdNet is a fully connected forward net agent
###
class FCFFwdUnsupAgent(nn.Module):
    def __init__(self, MetaParams, HyperParams):
        super(FCFFwdUnsupAgent, self).__init__()
        self.MetaParams = MetaParams
        self.HyperParams = HyperParams

        self.Depth = int(HyperParams["Depth"])
        self.InWidth = int(HyperParams["InWidth"])
        self.OutWidth = int(HyperParams["OutWidth"])
        self.HiddenWidths = HyperParams["HiddenWidth"]
        if type(self.HiddenWidths) != list:
            self.HiddenWidths = [int(self.HiddenWidths)] * (self.Depth - 1)
        self.gHidden = HyperParams["gHidden"]
        self.gLast = HyperParams["gLast"]

        self.Activations = [None]*(self.Depth+1)
        self.UseBias = False

        self.LinLayers = [None] * (self.Depth)
        if self.Depth > 2:
            self.LinLayers = nn.ModuleList([nn.Linear(
                self.HiddenWidths[i], self.HiddenWidths[i + 1], bias=self.UseBias) for i in range(self.Depth - 2)])
            self.LinLayers.insert(0, nn.Linear(self.InWidth, self.HiddenWidths[0], bias=self.UseBias))
            self.LinLayers.append(nn.Linear(self.HiddenWidths[-1], self.OutWidth, bias=self.UseBias))
        elif self.Depth == 2:
            self.LinLayers = nn.ModuleList([nn.Linear(self.InWidth, self.HiddenWidths[0], bias=self.UseBias),
                                            nn.Linear(self.HiddenWidths[0], self.OutWidth, bias=self.UseBias)])
        elif self.Depth == 1:
            self.LinLayers = nn.ModuleList([nn.Linear(self.InWidth, self.OutWidth, bias=self.UseBias)])
        else:
            raise Exception("Depth must be a positive integer.")

        # Better initialization than the default
        for i in range(self.Depth):
            #torch.nn.init.xavier_uniform_(self.LinLayers[i].weight, gain=math.sqrt(2.0))
            torch.nn.init.kaiming_uniform_(self.LinLayers[i].weight, nonlinearity='tanh')

    # Forward pass
    def forward(self, x):
        self.Activations[0] = nn.Flatten()(x)
        for i in range(self.Depth-1):
            self.Activations[i+1] = self.gHidden(self.LinLayers[i](self.Activations[i]))
        self.Activations[self.Depth] = self.gLast(self.LinLayers[self.Depth-1](self.Activations[self.Depth-1]))
        return self.Activations[self.Depth]


    def InitializeParams(self):
        for i in range(self.Depth):
            torch.nn.init.xavier_uniform(self.LinLayers[i].weight)
            if self.UseBias:
                self.LinLayers[i].weight.bias.data.fill_(0.01)

    # Function for updating weights
    def UpdateParams(self):
        for i in range(self.Depth):

            #dW=torch.zeros_like(self.LinLayers[i].weight)
            W=self.LinLayers[i].weight.data
            aPre=self.Activations[i]
            aPost=self.Activations[i+1]

            MeanaPre = aPre.mean(dim=0)
            MeanaPost = aPost.mean(dim=0)

            # These are used for scaling updates rules.
            # std of entries in W
            Wscale=math.sqrt(1/W.shape[1])
            # sqrt of pre and post sizes
            PreScale=math.sqrt(W.shape[1])
            PostScale=math.sqrt(W.shape[0])
            # batch size
            BatchScale=len(aPre)

            #print('ps',i,Wscale**2,PreScale,PostScale,BatchScale)
            #print('map',MeanaPost.var().item(),1/math.sqrt(BatchScale))
            # print('dw',i,W.var().item(),
            #       (MeanaPost[:,None].expand(W.shape)*Wscale*PreScale).var().item(),
            #       (aPost.T@aPre*PreScale*Wscale/BatchScale).var().item(),
            #       (aPost.T@(aPre-aPost@W)*PreScale*Wscale/BatchScale).var().item()
            #       )


            # Weight decay
            W-=self.MetaParams[1]*W

            # New
            W-=self.MetaParams[3]*MeanaPost[:,None].expand(W.shape)*Wscale*PreScale

            # Pure Hebbian
            W-=self.MetaParams[4]*aPost.T@aPre*PreScale*Wscale/BatchScale

            # Homeostatic
            #W+=self.MetaParams[5]*(1.0-aPost.T)@aPre
            #W+=self.MetaParams[5]*W*MeanaPre
            #W+=self.MetaParams[6]*W*MeanaPost[:, torch.newaxis]

            # Oja's
            W+=self.MetaParams[7]*aPost.T@(aPre-aPost@W)*PreScale*Wscale/BatchScale




