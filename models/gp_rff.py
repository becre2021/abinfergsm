from models_utility.param_gp import Param
from models_utility.spt_manager_train import spt_manager_train
from models_utility.function_gp import cholesky, lt_log_determinant
from torch import triangular_solve
from models_utility.likelihoods import Gaussian

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

torch.set_default_tensor_type(torch.DoubleTensor)

zitter = 1e-8
class ssgpr_sm(nn.Module):    
    
    def __init__(self, num_batch, num_sample_pt, param_dict, kernel=None, likelihood=None, device = None ):
        super(ssgpr_sm,self).__init__()        
        self.device = torch.device("cuda") if device else torch.device('cpu')
        self.name = None
        self.num_batch = num_batch
        self.num_samplept = num_sample_pt
        self._set_up_param(param_dict)
        self.total_num_sample = self.num_samplept*self.weight.numel()
        self.likelihood = Gaussian(variance= self.noise, device = device) if likelihood == None else likelihood

        return 
        
        
    def _get_param_list(self):
        weight_list,mu_list = [],[],[]
        if hasattr(self,'kernel') and self.kernel is not None:
            weight_list.append(self.kernel.weight.transform().cpu().data.numpy())
            mu_list.append(self.kernel.mu.transform().cpu().data.numpy())
            std_list.append(self.kernel.std.transform().cpu().data.numpy())
        else:
            weight_list.append(self.weight.transform().cpu().data.numpy())
            mu_list.append(self.mu.transform().cpu().data.numpy())
            std_list.append(self.std.transform().cpu().data.numpy())

        return weight_list,mu_list,std_list


    def _set_up_param(self, param_dict):
        self.input_dim = param_dict['input_dim']
        self.num_Q = param_dict['Num_Q']
        self.noise = param_dict['noise_err']
        self.lr_hyp = param_dict['lr_hyp']
        self.sampling_option = 'equal'


        hypparam_dict = param_dict['hypparam']
        self.sf2 = Param(torch.tensor(hypparam_dict['noise_variance']).to(self.device),
                         requires_grad=False, requires_transform=True , param_name='sf2')

        self.weight = Param(torch.tensor(hypparam_dict['weight']).view(-1,1).to(self.device) ,
                            requires_grad=True, requires_transform=True , param_name='weight')
        self.mu = Param(torch.tensor(hypparam_dict['mean'] ).view(-1,self.input_dim).to(self.device),
                        requires_grad=True, requires_transform=True, param_name='mu')
        self.std = Param(torch.tensor(hypparam_dict['std']).view(-1, self.input_dim).to(self.device),
                         requires_grad=True,requires_transform=True, param_name='std')

        self.noise_rv = [Variable(torch.randn(self.num_samplept, self.input_dim).to(self.device),requires_grad=False) for i in range(self.num_Q)]
        self._set_spectral_pt()

        return

    def _set_data(self,x_train, y_train):
        self.x = x_train
        self.y = y_train
        return 
    

    def _set_spectral_pt(self):
        sampled_spectral_pt_list = []
        for i_th in range(self.num_Q):
            #print(self.noise_rv[i_th].requires_grad)
            sampled_spectal_pt = self.mu.transform()[i_th] + self.std.transform()[i_th].mul(self.noise_rv[i_th])
            sampled_spectral_pt_list.append(sampled_spectal_pt)
        self.sampled_spectral_pt = sampled_spectral_pt_list
        return 

    
    def _get_param(self):
        for ith in self.parameters():
            print('%s : %s'%(ith.param_name,ith.transform()))


    def _sampling_gaussian(self, mu, std, num_sample):
        eps = Variable(torch.randn(num_sample, self.input_dim).to(self.device))
        return mu + std.mul(eps)
        # return mu



    def _compute_gaussian_basis(self, x, xstar=None):
        sampled_spectral_pt = self._sampling_gaussian(self.mu.transform(),
                                                  self.std.transform(),
                                                  self.num_sample_pt)  # self.num_sample x dim
        xdotspectral = x.matmul(sampled_spectral_pt.t())
        Phi = torch.cat([xdotspectral.cos(), xdotspectral.sin()], 1).to(self.device)
        if xstar is None:
            return Phi
        else:
            xstardotspectral = xstar.matmul(sampled_spectral_pt.t())
            Phi_star = torch.cat([xstardotspectral.cos(), xstardotspectral.sin()], 1).to(self.device)
            return Phi, Phi_star




    def _compute_sm_basis(self, x, xstar=None):
        multiple_Phi = []
        current_sampled_spectral_list = []
        if self.weight.shape[0] > 1:
            current_pi = self.weight.transform().reshape([1, -1]).squeeze()
        else:
            current_pi = self.weight.transform()

        self._set_spectral_pt()
        for i_th in range(self.weight.numel()):
            sampled_spectral_pt = self.sampled_spectral_pt[i_th]
            if xstar is not None:
                current_sampled_spectral_list.append(sampled_spectral_pt)
            xdotspectral = (2 * np.pi) * x.matmul(sampled_spectral_pt.t())


            Phi_i_th = (current_pi[i_th] / self.num_samplept).sqrt() * torch.cat([xdotspectral.cos(), xdotspectral.sin()], 1).to(self.device)
            multiple_Phi.append(Phi_i_th)

        if xstar is None:
            return torch.cat(multiple_Phi, 1)
        else:
            multiple_Phi_star = []
            for i_th, current_sampled in enumerate(current_sampled_spectral_list):
                xstardotspectral = (2 * np.pi) * xstar.matmul(current_sampled.t())

                Phistar_i_th = (current_pi[i_th] / self.num_samplept).sqrt() * torch.cat([xstardotspectral.cos(), xstardotspectral.sin()],1).to(self.device)
                multiple_Phi_star.append(Phistar_i_th)
            return torch.cat(multiple_Phi, 1), torch.cat(multiple_Phi_star, 1)



    def _compute_gram_approximate(self, Phi):
        return  Phi.t().matmul(Phi) + (self.likelihood.variance.transform()**2 + zitter).expand(Phi.shape[1], Phi.shape[1]).diag().diag()



    def _compute_kernel_sm_approximate(self, x ,  normalized_option = False):
        Phi_list = self._compute_sm_basis(x)
        kernel_output = self.sf2.transform()* Phi_list.matmul(Phi_list.t())
        if normalized_option == True:
            return kernel_output/kernel_output[0,0]
        else:
            return kernel_output




    def compute_loss(self,batch_x,batch_y,kl_option ,current_iter = 1):
        """
        :param batch_x:
        :param batch_y:
        :return: approximate lower bound of negative log marginal likelihood
        """


        num_input = batch_x.shape[0]

        # negative logmarginal likelihood
        loss = 0
        for j_th in range(self.num_batch):
            Phi = self._compute_sm_basis(batch_x)
            Approximate_gram = self._compute_gram_approximate(Phi)
            L = cholesky(Approximate_gram)
            Lt_inv_Phi_y = triangular_solve((Phi.t()).matmul(batch_y), L ,upper=False)[0]  # trtrs(tri_matrix, rhs, lower=True):            
            loss += (0.5 / self.likelihood.variance.transform()**2) * (batch_y.pow(2).sum() - Lt_inv_Phi_y.pow(2).sum())
            loss += lt_log_determinant(L)
            loss += (-self.total_num_sample)*2* self.likelihood.variance
            loss += 0.5 * num_input * (np.log(2*np.pi) + 2*self.likelihood.variance )
            
        return (1 / self.num_batch) * loss

        

    def _predict(self, inputs_new, diag = True):
        if isinstance(inputs_new, np.ndarray):
            inputs_new = Variable(torch.Tensor(inputs_new).to(self.device), requires_grad=False)

        Phi, Phi_star = self._compute_sm_basis(self.x, inputs_new)
        Approximate_gram = self._compute_gram_approximate(Phi)

        L = cholesky(Approximate_gram)
        Lt_inv_Phi_y = triangular_solve((Phi.t()).matmul(self.y),L ,upper= False )[0]           
        alpha =  triangular_solve(Lt_inv_Phi_y,L.t(), upper=True)[0]

        mean_f = Phi_star.matmul(alpha)        
        #Lt_inv_Phistar_t = trtrs(L, Phi_star.t(), lower=True).t()
        Lt_inv_Phistar_t = triangular_solve(Phi_star.t(),L,upper=False)[0].t()
        
        
        if diag:
            mean_var = (self.likelihood.variance.transform()**2)*(1 + Lt_inv_Phistar_t.pow(2).sum(1))
            mean_var = mean_var.reshape(-1, 1)
        else:
            mean_var = (self.likelihood.variance.transform()**2)*(1 + Lt_inv_Phistar_t.matmul(Lt_inv_Phistar_t.t()))
        return mean_f, mean_var


    
    def _eval_K(self,x1, x2=None):
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



    def _predict_exact(self, inputs_new, diag=True):
        if isinstance(inputs_new, np.ndarray):
            inputs_new = torch.Tensor(inputs_new).to(self.device)
        num_input = self.x.shape[0]


        kxx = self._eval_K(self.x.detach()) + (zitter + self.likelihood.variance.transform()**2).expand(num_input, num_input).diag().diag()        
        k_xs = self._eval_K(self.x.detach(), inputs_new.detach())

        # reference_code
        L = cholesky(kxx)
        A = triangular_solve(k_xs,L,upper=False)[0]
        V = triangular_solve(self.y,L,upper=False)[0]
        
        pred_mean = torch.mm(torch.transpose(A, 0, 1), V)
        if diag:
            var_f1 = self._eval_K(inputs_new).diag()
            var_f2 = torch.sum(A * A, 0)
            return pred_mean, (var_f1 - var_f2).reshape(-1, 1) +  self.likelihood.variance.transform()**2
        else:
            var_f1 = self._eval_K(inputs_new)
            var_f2 = torch.mm(A.t(), A)
            return pred_mean, (var_f1 - var_f2) +  (self.likelihood.variance.transform()**2).diag().diag()
    
