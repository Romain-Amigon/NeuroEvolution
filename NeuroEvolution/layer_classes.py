# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 00:12:46 2025

@author: Romain
"""


from dataclasses import dataclass
from typing import Type
import torch.nn as nn

@dataclass
class LinearCfg:
    in_features: int
    out_features: int
    activation: Type[nn.Module]

@dataclass
class Conv2dCfg:
    in_channels: int
    out_channels: int
    kernel_size: int | tuple
    stride: int = 1
    padding: int = 0
    activation: Type[nn.Module] = nn.ReLU

@dataclass
class DropoutCfg:
    p: float

@dataclass
class FlattenCfg:
    start_dim: int = 1
