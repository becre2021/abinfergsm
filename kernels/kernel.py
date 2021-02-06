
#from ..models_utility.param_gp import Param

# import sys
# sys.path.append('..')
from models_utility.param_gp import Param


import torch
import torch.nn as nn
import numpy as np
TensorType = torch.DoubleTensor



class Kernel(nn.Module):
    def __init__(self):
        super(Kernel, self).__init__()

    def check_tensortype(self, x1, x2=None):
        if torch.is_tensor(x1) == False:
            x1 = TensorType(x1)
        if x2 is None:
            return x1, x1
        else:
            if torch.is_tensor(x2) == False:
                x2 = TensorType(x2)
            return x1, x2


    def K(self, x1, x2 = None):
        return NotImplementedError

    def K_diag(self, x1, x2=None):
        return torch.diag(self.K(x1, x2))




class StationaryKernel(Kernel):
    def __init__(self, device ):
        super(StationaryKernel, self).__init__()
        self.ARD = False
        self.input_dim = None
        self.device = torch.device("cuda") if device else torch.device('cpu')


    def _sq_dist(self,x1,x2 = None):
        x1,x2 = self.check_tensortype(x1,x2)

        x1_ = x1.pow(2).sum(-1,keepdim = True)
        x2_ = x2.pow(2).sum(-1,keepdim = True)

        #sq_dist = x1.matmul(x2.transpose(-2,-1)).mul_(-2).add_(x2_.transpose(-2,-1)).add_(x1_)
        sq_dist = -2*x1.matmul(x2.transpose(-2, -1)) + (x1_ + x2_.transpose(-2, -1))
        sq_dist.clamp_min(1e-16)

        return sq_dist



    def K(self, x1, x2 = None):
        return NotImplementedError


    def K_diag(self, x1, x2=None):
        return torch.diag(self.K(x1, x2))

