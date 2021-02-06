import torch
from torch import nn
from models_utility.param_gp import Param



#class Gaussian(nn.Module):
class Gaussian(object):    
    def __init__(self, variance = None, device = None):
        self.device = torch.device("cuda") if device else torch.device('cpu')
        
        # std
        self.variance = Param(torch.tensor([variance]).to(self.device), requires_grad = True, requires_transform=True , param_name= 'noise_variance')
        
    def log_p(slef,F,Y):
        #return densities.gaussian(F,Y,self.variance)
        return

    def predict_mean_variacne(self, mean_f, var_f):
        return mean_f, var_f + self.variance.transform().expand_as(var_f)

    def predict_mean_covariance(self, mean_f, var_f):
        return mean_f, var_f + self.variance.transform().expand_as(var_f).diag().diag()



if __name__ == "__main__":
    print(1)
