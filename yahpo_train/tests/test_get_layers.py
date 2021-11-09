
import pytest
import torch
import numpy as np
from yahpo_train.cont_scalers import *
from yahpo_train.model import SigmoidRange
import pandas as pd



def get_arch(max_units, n, shape):
    if max_units == 0:
        return []
    if shape == "square":
        return [2**max_units for x in range(n)]
    if shape == "cone":
        units = [2**max_units]
        for x in range(n):
            units += [units[-1]/2]
        return units
            
            
if __name__ == '__main__':
    print(get_arch(0, 1, "conic"))
    print(get_arch(7, 1, "conic"))
    print(get_arch(7, 3, "conic"))
    print(get_arch(7, 4, "square"))
    