#from __future__ import absolute_import

from models_utility.param_gp import Param
from models_utility.function_gp import  cholesky, lt_log_determinant
from models_utility.likelihoods import Gaussian
from torch import triangular_solve


from models.gp import gpr
import numpy as np
import torch
import torch.nn as nn


import torch
from torch.autograd import Variable
import numpy as np


jitter = 1e-8
class gp_vfe(gpr):

    def __init__(self, kernel, likelihood , device, param_dict):
        super(gp_vfe,self).__init__(kernel, likelihood, device ,param_dict)
        self.device = torch.device("cuda") if device else torch.device('cpu')
        #self.jitter = torch.tensor(1e-8).to(self.device)
        self.x = None
        self.y = None


    def _set_data(self, batch_x, batch_y):
        self.x = batch_x
        self.y = batch_y
        return

    def _set_inducing_pt(self,num_inducing):
        assert(self.x is not None)
        self.num_inducing_pt = num_inducing
        idx = np.random.choice(len(self.x),num_inducing)
        self.inducing_pt = Param(torch.tensor(self.x[idx]).to(self.device), requires_grad=True, requires_transform=False, param_name = 'inducing_pt')
        return

    def compute_loss(self,batch_x,batch_y,kl_option):
        num_inducing_pt = self.num_inducing_pt
        dim_output = batch_y.shape[1]
        num_data = batch_x.shape[0]

        K_ff_diag = self._compute_Kxx_diag(self.x)
        K_uf = self._compute_Kxs(self.inducing_pt, self.x)
        K_uu = self._compute_Kxx(self.inducing_pt)

        L = cholesky(K_uu)
        A = triangular_solve(K_uf, L, upper=False)[0]
        A_At = A.matmul(A.t()).div(self.likelihood.variance.transform()**2)
        B = A_At + torch.eye(int(num_inducing_pt)).to(self.device)
        L_B = cholesky(B)
        C = triangular_solve(A.matmul(self.y), L_B, upper=False)[0].div(self.likelihood.variance.transform()**2)

        Elbo = 0.
        Elbo += torch.tensor([-0.5 * dim_output * num_data * np.log(2 * np.pi)]).to(self.device)
        Elbo -= dim_output * L_B.diag().log().sum()
        Elbo -= 0.5 * dim_output * num_data * 2*self.likelihood.variance
        Elbo -= (0.5 / (self.likelihood.variance.transform()**2)) * (self.y.pow(2).sum() + dim_output * K_ff_diag.sum())
        Elbo += 0.5 * C.pow(2).sum()
        Elbo += 0.5 * dim_output * A_At.diag().sum()

        return -Elbo.squeeze()



    def _predict(self, inputs_new, diag = True):
        if isinstance(inputs_new, np.ndarray):
            inputs_new = torch.Tensor(inputs_new).to(self.device)

        num_inducing_pt = self.num_inducing_pt
        dim_output = self.y.shape[1]

        K_ff_diag = self._compute_Kxx_diag(self.x)
        K_uf = self._compute_Kxs(self.inducing_pt, self.x)
        K_uu = self._compute_Kxx(self.inducing_pt)
        K_us = self._compute_Kxs(self.inducing_pt, inputs_new)

        L = cholesky(K_uu)
        A = triangular_solve(K_uf, L, upper=False)[0]
        A_At = A.matmul(A.t()).div(self.likelihood.variance.transform()**2)
        B = A_At + torch.eye(int(num_inducing_pt)).to(self.device)
        L_B = cholesky(B)
        C = triangular_solve(A.matmul(self.y), L_B, upper=False)[0].div(self.likelihood.variance.transform()**2)

        temp1 = triangular_solve(K_us, L, upper=False)[0]
        temp2 = triangular_solve(temp1, L_B, upper=False)[0]
        mean_f = temp2.t().matmul(C)

        if diag:
            var = self._compute_Kxx_diag(inputs_new) + temp2.t().pow(2).sum(1) - temp1.pow(2).sum(0).squeeze() +  (self.likelihood.variance.transform()**2).diag().diag()
        else:
            var = self._compute_Kxx(inputs_new) + temp2.t().matmul(temp2) - temp1.t().matmul(temp1) +  self.likelihood.variance.transform()**2  
        return mean_f, var
    
              
        
    

