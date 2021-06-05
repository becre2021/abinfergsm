from models.gp_rrff import ssgpr_rep_sm
from models_utility.param_gp import Param
from models_utility.spt_manager_train import spt_manager_train
from models_utility.function_gp import cholesky, lt_log_determinant
from torch import triangular_solve
from models_utility.likelihoods import Gaussian
from models_utility.personalized_adam import Adam_variation


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions import kl_divergence




#torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)



zitter = 1e-8
pi2t = torch.tensor([2*np.pi])
class ssgpr_rep_sm_reg(ssgpr_rep_sm):
    def __init__(self, num_batch, num_sample_pt, param_dict, kernel = None, likelihood=None, device = None):
        super(ssgpr_rep_sm_reg, self).__init__(num_batch = num_batch,
                                                num_sample_pt = num_sample_pt,
                                                param_dict = param_dict,
                                                kernel=kernel,
                                                likelihood=likelihood,
                                                device = device)
        self.name = None        
        self.likelihood = Gaussian(variance= self.noise, device = device) if likelihood == None else likelihood        
        self._set_up_param(param_dict)
        self.spt_manager = spt_manager_train(spt = num_sample_pt,  num_Q = self.num_Q ,rate= param_dict['weight_rate'])      
        self.num_samplept_list_at = None
        self.alpha = []
        print('total spt:{}, spt:{}, Q:{} in setup model '.format(self.total_num_sample,self.num_samplept,self.num_Q))

        

    def _set_up_param(self, param_dict):
        self.input_dim = param_dict['input_dim']
        self.num_Q = param_dict['num_Q']
        self.noise = param_dict['noise_err']
        self.lr_hyp = param_dict['lr_hyp']
        self.sampling_option = param_dict['sampling_option']


        hypparam_dict = param_dict['hypparam']
        self.sf2 = Param(torch.tensor(hypparam_dict['noise_variance']).to(self.device),
                         requires_grad=False, requires_transform=True , param_name='sf2')

        self.weight = Param(torch.tensor(hypparam_dict['weight']).view(-1,1).to(self.device) ,
                            requires_grad=True, requires_transform=True , param_name='weight')
        self.std = Param(torch.tensor(hypparam_dict['std']).view(-1, self.input_dim).to(self.device),
                         requires_grad=True, requires_transform=True, param_name='std')

        self.mu = Param(torch.tensor(hypparam_dict['mean'] ).view(-1,self.input_dim).to(self.device),
                        requires_grad=True, requires_transform=True, param_name='mu')

        self.std_prior = Param(torch.tensor(hypparam_dict['std_prior']).view(-1,self.input_dim).to(self.device),
                               requires_grad=False, requires_transform=True , param_name='std_prior')

        self.mu_prior = Param(torch.tensor(hypparam_dict['mean_prior']).view(-1,self.input_dim).to(self.device) ,
                              requires_grad=False, requires_transform=True , param_name='mu_prior')
        
        return



    def _set_num_spectralpt(self,num_pt,intrain =True):
        self.num_samplept = num_pt
        self.total_num_sample = self.num_samplept*self.weight.numel()
        self.spt_manager._set_num_spectralpt(num_pt,intrain) 
        #print('total spt {}, spt {}'.format(self.total_num_sample,self.num_sampleplt))        
        return 
    


    def _assign_num_spectralpt(self,x,intrain = True):        
        if self.sampling_option == 'weight':                
            assigned_spt, ratio =  self.spt_manager.calc_sptratio_given_X( weight_param = self.weight.transform(),
                                                                           mu_param = self.mu.transform(),
                                                                           std_param = self.std.transform(),
                                                                           X = x,
                                                                           intrain = intrain)
            outs = list(assigned_spt)

        elif self.sampling_option == 'naive_weight':
            assigned_spt, ratio =  self.spt_manager.calc_sptratio_naive(weight_param_log = self.weight, 
                                                                        intrain = intrain)
            outs = list(assigned_spt)

        else:
            outs = [self.num_samplept for ith in range(self.num_Q)]

        self.num_samplept_list_at = outs    
        return outs
    
    
    
    
    def _kl_div_qp(self):
        q_dist = MVN(loc = self.mu.transform().view(1, -1).squeeze() ,
                     covariance_matrix = self.std.transform().view(1,-1).squeeze().pow(2).diag() )

        p_dist = MVN(loc = self.mu_prior.transform().view(1, -1).squeeze(),
                     covariance_matrix = self.std_prior.transform().view(1,-1).squeeze().pow(2).diag() )

        return kl_divergence(q_dist, p_dist)

    
        
        
    def _compute_gram_approximate(self, Phi):
        return  Phi.t().matmul(Phi) + (self.likelihood.variance.transform()**2 + zitter).expand(Phi.shape[1], Phi.shape[1]).diag().diag()
    
    
    def _compute_nlml(self,batch_x,batch_y,intrain = True):
        """
        :param batch_x:
        :param batch_y:
        :return: approximate lower bound of negative log marginal likelihood
        """


        num_input = batch_x.shape[0]
        loss = 0
        for j_th in range(self.num_batch):
            Phi = self._compute_sm_basis(batch_x, intrain = intrain)
            Approximate_gram = self._compute_gram_approximate(Phi)
            L = cholesky(Approximate_gram)
            Linv_PhiT = triangular_solve(Phi.t(), L ,upper=False)[0]           
            Linv_PhiT_y = Linv_PhiT.matmul(batch_y) 

            loss00 = (0.5 / self.likelihood.variance.transform()**2) * (batch_y.pow(2).sum() - Linv_PhiT_y.pow(2).sum())
            loss01 = lt_log_determinant(L)
            loss += loss00
            loss += loss01            
            loss += (-self.total_num_sample)* (2* self.likelihood.variance)
            loss += 0.5 * num_input * (np.log(2*np.pi) + 2*self.likelihood.variance )
            return loss
            
            
    def compute_loss(self,batch_x,batch_y,kl_option ,current_iter = 1):
        loss = self._compute_nlml(batch_x,batch_y)
        
        if kl_option == False:
            return (1 / self.num_batch) * loss
        else:
            kl_term = self._kl_div_qp()
            return (1 / self.num_batch) * loss + kl_term  
    
    
    
    def _predict_single(self, inputs_new, diag = True):
        if isinstance(inputs_new, np.ndarray):
            inputs_new = Variable(torch.Tensor(inputs_new).to(self.device), requires_grad=False)
        
        Phi, Phi_star = self._compute_sm_basis(self.x, inputs_new ,intrain = False)
        Approximate_gram = self._compute_gram_approximate(Phi)
        L = cholesky(Approximate_gram)
        Lt_inv_Phi_y = triangular_solve((Phi.t()).matmul(self.y),L ,upper= False )[0]           
        alpha =  triangular_solve(Lt_inv_Phi_y,L.t(), upper=True)[0]

        mean_f = Phi_star.matmul(alpha)        
        Lt_inv_Phistar_t = triangular_solve(Phi_star.t(),L,upper=False)[0].t()        
        
        if diag:
            mean_var = (self.likelihood.variance.transform()**2)*(1 + Lt_inv_Phistar_t.pow(2).sum(1))
            mean_var = mean_var.reshape(-1, 1)
        else:
            mean_var = (self.likelihood.variance.transform()**2)*(1 + Lt_inv_Phistar_t.matmul(Lt_inv_Phistar_t.t()))
        return mean_f, mean_var

    
    
    def _predict(self, inputs_new , num_sample = 3, diag = True):
        f_mean,f_var = 0,0
        for i in range(num_sample):
            i_f_mean,i_f_var = self._predict_single(inputs_new,diag)
            f_mean += i_f_mean
            f_var += i_f_var
        
        return f_mean/num_sample, f_var/num_sample
    
