import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearAE(nn.Module):
    def __init__(self,
                 input_dim_list = [784, 400], 
                 symmetry = True, 
                 device = "cuda",
                 dropout = 0.2,
                 bias = True):
        super(LinearAE, self).__init__()
        assert len(input_dim_list) >= 2
        self.input_dim = len(input_dim_list)
        self.symmetry = symmetry
        self.device = device
        self.dropout = nn.Dropout(p = dropout)
        self.encoder_layer_list = nn.ModuleList()
        self.bias = bias
        #initialize the weights from the encoder part
        for i in range(len(input_dim_list) - 1):
            self.encoder_layer_list.append(nn.Linear(in_features= input_dim_list[i],
                                                      out_features= input_dim_list[i+1],
                                                     bias = bias))
        #if the autoencoder does not enforce symmetry, we need to initialize the weights for the decoder part
        output_dim_list = input_dim_list[::-1]
        if not symmetry:
            self.decoder_layer_list = nn.ModuleList()
            for i in range(len(output_dim_list) -1 ):
                self.decoder_layer_list.append(nn.Linear(in_features= output_dim_list[i],
                                                      out_features= output_dim_list[i+1],
                                                        bias = bias))
        elif symmetry and self.bias:
            #initialize the decoder bias
            self.bias_list = []
            for i in range(len(output_dim_list) -1):
                self.bias_list.append(torch.nn.Parameter(torch.rand(output_dim_list[i+1])).to(self.device))


    def forward(self, x):
        weight_list = []
        for i in range(self.input_dim - 1):
            x = self.encoder_layer_list[i](x)
            weight_list.append(self.encoder_layer_list[i].weight)
            x = torch.relu(x)
            x = self.dropout(x)
        if self.symmetry:
            weight_list = weight_list[::-1]
            for i in range(len(weight_list)-1):
                x = F.linear(x, torch.t(weight_list[i])) + self.bias_list[i] if self.bias else F.linear(x, torch.t(weight_list[i]))
                x = torch.relu(x)
                x = self.dropout(x)
            return torch.sigmoid(F.linear(x, torch.t(weight_list[-1])) + self.bias_list[-1])
        elif not self.symmetry:
            for i in range(len(self.decoder_layer_list)-1):
                x = self.decoder_layer_list[i](x)
                x = torch.relu(x)
                x = self.dropout(x)
            return torch.sigmoid(self.decoder_layer_list[-1](x))

        