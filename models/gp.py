#from __future__ import absolute_import

from models_utility.param_gp import Param
from models_utility.function_gp import cholesky, lt_log_determinant
from models_utility.likelihoods import Gaussian
from torch import triangular_solve
#from torch import triangular_solve as trtrs


import numpy as np
import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)
zitter = 1e-8
class gpmodel(nn.Module):
    def __init__(self, kernel, likelihood , device ):
        super(gpmodel,self).__init__()
        self.kernel = kernel
        self.device = torch.device("cuda") if device else torch.device('cpu')
        self.likelihood = Gaussian(variance= [1.0], device = device) if likelihood == None else likelihood
        self.zitter = torch.tensor(zitter).to(self.device)


    def compute_loss(self):
        raise NotImplementedError

    def _compute_Kxx(self,x):
        num_input = x.shape[0]
        return self.kernel.K(x) 

    def _compute_Kxx_diag(self,x):
        #return self._compute_Kxx(x).diag()
        return self._compute_Kxx(x).diag()

    def _compute_Kxs(self,x,xstar):
        return self.kernel.K(x,xstar)
    


class gpr(gpmodel):
    def __init__(self, kernel, likelihood, device, param_dict ):
        super(gpr,self).__init__(kernel, likelihood, device )
        self.lr_hyp = param_dict['lr_hyp']
        self.likelihood = Gaussian(variance= param_dict['noise_err'], device = device) if likelihood == None else likelihood
        self.name = None

        
    def _set_data(self,batch_x, batch_y ):
        self.x = batch_x
        self.y = batch_y

        
    def _get_param_list(self):
        weight_list = []
        mu_list = []
        std_list = []
        if hasattr(self,'kernel') and self.kernel is not None:
            weight_list.append(self.kernel.weight.transform().cpu().data.numpy())
            mu_list.append(self.kernel.mu.transform().cpu().data.numpy())
            std_list.append(self.kernel.std.transform().cpu().data.numpy())

        else:
            weight_list.append(self.weight.transform().cpu().data.numpy())
            mu_list.append(self.mu.transform().cpu().data.numpy())
            std_list.append(self.std.transform().cpu().data.numpy())

        return weight_list,mu_list,std_list



    def compute_loss(self, batch_x, batch_y , kl_option ):
        self._set_data(batch_x, batch_y)
        num_input,dim_output = batch_y.shape
        gram_matrix = self._compute_Kxx(batch_x) + (zitter + self.likelihood.variance.transform()**2).expand(num_input,num_input).diag().diag()  
        L = cholesky(gram_matrix)
        alpha = triangular_solve(batch_y,L,upper=False)[0]        
        
        loss = 0.5 * alpha.pow(2).sum() + lt_log_determinant(L) + 0.5 * num_input * np.log(2.0 * np.pi)
        return loss    

    
    def _predict(self, inputs_new,  diag = True):
        if isinstance(inputs_new, np.ndarray):
            inputs_new = torch.Tensor(inputs_new).type(self.tensor_type)

        num_input = self.x.shape[0]
        k_xx = self._compute_Kxx(self.x.detach()) + (zitter + self.likelihood.variance.transform()**2).expand(num_input, num_input).diag().diag()        
        k_xs = self._compute_Kxs(self.x,inputs_new)

        L = cholesky(k_xx)
        A = triangular_solve(k_xs,L,upper=False)[0]   #A = trtrs(L, k_xs)
        V = triangular_solve(self.y,L,upper=False)[0]

        mean_f = torch.mm(torch.transpose(A, 0, 1), V)        
        if diag:
            var_f1 = self.kernel.K_diag(inputs_new)
            var_f2 = torch.sum(A * A, 0)
            return mean_f, (var_f1 - var_f2).reshape(-1,1) +  self.likelihood.variance.transform()**2

        else:
            var_f1 = self.kernel.K(inputs_new)
            var_f2 = torch.mm(A.t(), A)
            return mean_f, (var_f1 - var_f2) +  (self.likelihood.variance.transform()**2).diag().diag()


        


