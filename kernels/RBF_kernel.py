
import torch
import numpy as np
from kernels.kernel import StationaryKernel
from models_utility.param_gp import Param

TensorType = torch.DoubleTensor




class RBF(StationaryKernel):

    def __init__(self,variance,length_scale,device):
        super(RBF, self).__init__(device)
        self._assign_RBF_param(variance,length_scale)

    def _assign_RBF_param(self,variance,length_scale):

        #self.input_dim = len(length_scale)
        self.variance = Param(torch.tensor(variance).to(self.device),
                              requires_grad=True, requires_transform=True ,param_name='rbf_variance' )

        self.length_scales = Param(torch.tensor(length_scale).to(self.device),
                                   requires_grad=True, requires_transform=True ,param_name='rbf_length')
        return


    def K(self, x1, x2 = None):
        x1, x2 = self.check_tensortype(x1, x2)
        x1 = x1.div(self.length_scales.transform())
        x2 = x2.div(self.length_scales.transform())
        r2 = self._sq_dist(x1, x2)

        return self.variance.transform()*torch.exp(-0.5 * r2)



if __name__ == "__main__":

    device = True
    variance = 1.0

    # x = torch.tensor(np.arange(0,3,0.5).reshape(-1,1)).type(TensorType)
    # length_scales = [1.0]

    x = torch.tensor(np.arange(0,3,0.5).reshape(-1,3)).type(TensorType)
    length_scales = [1.0,2.0,3.0]

    #print(x)
    #a = x.pow(2).sum(-1, keepdim=True)
    #print(a + a.transpose(-2,1) -2*x.matmul(x.transpose(-2, -1)))
    #-2 * x1.matmul(x2.transpose(-2, -1))

    Kern = RBF(variance,length_scales,device)
    Kern.K(x)


    # for ith in Kern.parameters():
    #     print(ith)