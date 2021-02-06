from models_utility.function_gp import SoftplusInv
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn import functional as F



class Param(Parameter):
    def __new__(cls, data=None, param_name= None,requires_grad=True, requires_transform=False, requires_clipping=False, clipping_range=False):

        if requires_transform:
            data = Param._transform_log(data, forward=False)
        return super(Param, cls).__new__(cls, data.double(), requires_grad=requires_grad)


    def __init__(self, data=None, param_name = None, requires_grad=True, requires_transform=False, requires_clipping=False, clipping_range=False):
        self.requires_transform = requires_transform
        self.requires_clipping = requires_clipping
        if requires_clipping == True:
            self.clipping_range = clipping_range
        self.prior = None
        self.param_name = param_name


    def transform(self):
        # Avoid in-place operation for Variable, using clone method  ???
        if self.requires_transform:
            return self._transform_log(self.clone(), forward=True)
        else:
            return self

    def __repr__(self):
        return self.param_name + ' : ' + self.data.__repr__()


    @staticmethod
    def _transform_log(x, forward):
        if forward:
            return torch.exp(x)
        else:
            return torch.log(x)

    @staticmethod
    def _trasform_softplus(x, forward):
        if forward:
            return F.softplus(x, threshold=35)
        else:
            return SoftplusInv(x)



if __name__ == "__main__":

    from models_utility.personalized_adam import Adam_variation

    mean = torch.from_numpy(np.array([1.,2.,3.]).reshape(-1,1))
    param_x = Param(torch.tensor(mean),requires_grad=True, requires_transform=True, param_name='mu')

    optimizer = Adam_variation([param_x],
                         lr=.001,
                         betas=(0.9, 0.999),
                         eps=1e-08,
                         weight_decay=0.0)


    for i in range(100 + 1):
        sum_square_y = (param_x.transform()).pow(2).sum()
        sum_square_y.backward()
        param_x.grad.data = param_x.grad.data/param_x.transform().data
        optimizer.step()
        optimizer.zero_grad()

        print(sum_square_y.data)


