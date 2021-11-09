
import pytest
import torch
import numpy as np
from yahpo_train.cont_scalers import *
from yahpo_train.model import SigmoidRange
import pandas as pd


def test_sigmoid_range():
    xs = [pd.Series([1,2,3]), pd.Series([-1,2,3]), pd.Series([-1,2,-3]), pd.Series([-1,-2,3])]
    for x in xs:
        sr = SigmoidRange(x)
        
        assert type(sr) == SigmoidRange
        assert (sr.high, sr.low) == (x.max(),x.min())
        
        tf = sr(torch.randn(100))
        assert torch.min(tf) >+ x.min()
        assert torch.max(tf) <= x.max()
    
    with pytest.raises(Exception):
        x = pd.Series([-1,-1,-1])
        SigmoidRange(x)
            
if __name__ == '__main__':
    test_sigmoid_range()
    y = torch.ones(1,3)
    y[0,0] = torch.Tensor([np.nan])
    z =torch.mean(torch.isnan(y).float())
    print(z)