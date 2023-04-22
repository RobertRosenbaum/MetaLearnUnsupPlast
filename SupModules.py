
import torch
import torch.nn as nn
import numpy as np


class FCFFwdSupModule(nn.Module):
    def __init__(self, HyperParams):
        super(FCFFwdSupModule, self).__init__()
        self.HyperParams = HyperParams

        self.Depth = int(HyperParams["Depth"])
        self.InWidth = int(HyperParams["InWidth"])
        self.OutWidth = int(HyperParams["OutWidth"])
        self.HiddenWidths = HyperParams["HiddenWidth"]
        if type(self.HiddenWidths) != list:
            self.HiddenWidths = [int(self.HiddenWidths)] * (self.Depth - 1)
        self.gHidden = HyperParams["gHidden"]
        self.gLast = HyperParams["gLast"]


        self.UseBias = True

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


    # Forward pass
    def forward(self, x):
        x = nn.Flatten()(x)
        for i in range(self.Depth-1):
            x = self.gHidden(self.LinLayers[i](x))
        x = self.gLast(self.LinLayers[self.Depth-1](x))
        return x
