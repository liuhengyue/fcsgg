"""
Swish Activation implementation.

Copied from https://github.com/sxhxliang/detectron2_backbone/blob/master/detectron2_backbone/layers/activations.py
"""
__author__ = "Shihua Liang"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Shihua Liang"]
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Shihua Liang"
__email__ = "sxhx.liang@gmail.com"

import torch
from torch import nn

__all__ = ["Swish", "MemoryEfficientSwish"]

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)