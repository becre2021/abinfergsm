import torch
import numpy as np
from kernels.kernel import StationaryKernel
from models_utility.param_gp import Param



class SM(StationaryKernel):
    def __init__(self,weight,mu,std,device):
        super(SM,self).__init__(device)
        self.device = torch.device("cuda") if device else torch.device('cpu')
        self._assign_SM_param(weight,mu,std)
        return


    def _assign_SM_param(self,weight,mu,std):
        self.num_Q = len(weight)
        self.input_dim = mu.shape[1]

        if self.input_dim > 1:
            self.ARD = True
        self.weight = Param(torch.tensor(weight).to(self.device), requires_grad=True, requires_transform=True, param_name='weight' )
        self.std = Param(torch.tensor(std).to(self.device), requires_grad=True, requires_transform=True, param_name='std' )
        self.mu = Param(torch.tensor(mu).to(self.device) , requires_grad=True, requires_transform=True, param_name='mu' )

        return


    def _init_SM_param_from_data(self,x,y):
        return


    def _creat_inputs_grid(self, x1_, x2_):
        return x1_.unsqueeze(-2), x2_.unsqueeze(-3)



    def K(self,x1, x2=None):
        if torch.is_tensor(x2):
            pass
        else:
            if x2 == None:
                x2 = x1

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        weight_, mu_, std_ = self.weight.transform(), self.mu.transform(), self.std.transform()

        out = 0
        for ith in range(self.num_Q):
            x1_ , x2_ = (2*np.pi)*x1.mul(std_[ith]), (2*np.pi)*x2.mul(std_[ith])
            sq_dist = -0.5*( -2 * x1_.matmul(x2_.t()) + (x1_.pow(2).sum(-1, keepdim=True) + x2_.pow(2).sum(-1, keepdim=True).t()))
            exp_term = sq_dist.exp()
            x11_, x22_ = (2 * np.pi) * x1.matmul(mu_[ith].reshape(-1,1)), (2 * np.pi) * x2.matmul(mu_[ith].reshape(-1,1))
            cos_term = (x11_- x22_.t() ).cos()
            out += weight_[ith]*exp_term.mul(cos_term)

        return out

    

    

if __name__ == "__main__":
    # 1d inputs
    x = torch.tensor(np.arange(0,1,0.1).reshape(-1,1))
    weight = np.random.rand(5,1)
    mu = np.random.rand(5, 1)
    std = np.random.rand(5, 1)


    # 2d inputs
    # x = torch.tensor(np.arange(0,1,0.1).reshape(-1,2))
    # weight = np.random.rand(5,1)
    # mu = np.random.rand(5, 2)
    # std = np.random.rand(5, 2)


    device = True
    Kern = SM(weight,mu,std,device)


    # for ith in Kern.parameters():
    #     print(ith)

    print(Kern.K(x))




    #print(Kern.weight)