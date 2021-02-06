import time
import torch
import random
import numpy as np
import math
from itertools import combinations
from scipy.special import softmax
import numpy as np

pi2 = 2 * math.pi
upper_bound_weight = 1e5
lower_bound_weight = 1e-8

def K_SM_Components(mu_list, std_list, x1, x2=None):
    if torch.is_tensor(x1):
        x1 = x1.cpu().data.numpy()
    if x2 == None:
        x2 = x1

    out = []
    for ith_mu, ith_std in zip(mu_list, std_list):
        x1_, x2_ = pi2 * (x1 * ith_std), pi2 * (x2 * ith_std)
        # sq_dist = -0.5*( -2 * x1_.matmul(x2_.t()) + (x1_.pow(2).sum(-1, keepdim=True) + x2_.pow(2).sum(-1, keepdim=True).t()))
        sq_dist = -0.5 * (-2 * np.matmul(x1_, x2_.T) + ((x1_ ** 2).sum(-1, keepdims=True) + (x2_ ** 2).sum(-1, keepdims=True).T))
        exp_term = np.exp(sq_dist)
        x11_, x22_ = pi2 * np.matmul(x1, ith_mu.reshape(-1, 1)), pi2 * np.matmul(x2, ith_mu.reshape(-1, 1))
        cos_term = np.cos(x11_ - x22_.T)
        out.append(exp_term * cos_term)

    return np.asarray(out)

    
    
    
class spt_manager_train(object):
    def __init__(self, spt, num_Q, rate=0.01, num_min_pt=2):            
        self.num_min_pt = int(spt / 2) + 1 if spt<=3 else int(spt / 2)            
        #self.num_min_pt = num_min_pt    
        
        self.spt = spt
        self.num_Q = num_Q
        self.total_spt = self.spt * self.num_Q        
        
        assert spt > self.num_min_pt
        self.rate = rate
        self.tau_set = None
        self.num_offdiag = None
        self.index_offdiag = None
        self.num_sample = None
        self.temperature = 1    
        self.call_num = 0
        return

    def _set_num_spectralpt(self, spt , num_min_pt=2 , intrain = True):
        if intrain : 
            self.num_min_pt = int(spt / 2) + 1 if spt<=3 else int(spt / 2)    
        else:
            self.num_min_pt = num_min_pt
        
        self.spt = spt
        assert spt > self.num_min_pt
        self.total_spt = self.spt * self.num_Q
        return

    def _set_collection_tauset(self, X):
        pass
        return

    def _get_subtauset(self, X):
        if torch.is_tensor(X):
            X = X.cpu().data.numpy()
        tau_set = (X[None, :, :] - X[:, None, :])
        num_train = X.shape[0]
        index_offdiag = np.triu_indices(num_train, k=1)
        return tau_set[index_offdiag]

    def k_sm(self, ith_weight, ith_mu, ith_std, tau_collection):
        exp_term_in = ((tau_collection * ith_std) ** 2).sum(axis=1, keepdims=True)
        exp_term = np.exp(-2 * (math.pi ** 2) * exp_term_in)
        cos_term_in = (tau_collection * ith_mu).sum(axis=1, keepdims=True)
        cos_term = np.cos(2 * math.pi * cos_term_in)
        return ith_weight * (exp_term * cos_term)

    def g_tau(self, mu, std, inputs):
        out = 1 + self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=2 * inputs)
        out += -2 * ((self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=inputs)) ** 2)
        return out

    def sum_g_tau(self, mu, std, inputs):
        return (self.g_tau(mu, std, inputs)).sum(axis=0)

    def h_taupair(self, mu, std, i_taus, j_taus, ij_taus_sum, ij_taus_minus):
        out = -self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=i_taus) * self.k_sm(ith_weight=1.0,
                                                                                                    ith_mu=mu,
                                                                                                    ith_std=std,
                                                                                                    tau_collection=j_taus)
        out += 0.5 * self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=ij_taus_sum)
        out += 0.5 * self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=ij_taus_minus)
        return out

    def sum_h_taupair(self, mu, std, i_taus, j_taus, ij_taus_sum, ij_taus_minus):
        return (self.h_taupair(mu, std, i_taus, j_taus, ij_taus_sum, ij_taus_minus)).sum(axis=0)

    
    
    def float_to_integer(self, ratio):
        num_minimum_total_pt = self.num_Q * self.num_min_pt
        num_allocated_total_pt = self.total_spt - num_minimum_total_pt
        assigned_spt_float = num_allocated_total_pt * ratio
        assigned_spt = np.asarray([int(np.round(ipt)) for ipt in assigned_spt_float])

        idx_plus = np.where((assigned_spt - assigned_spt_float) > 0.0)[0]
        idx_mius = np.where((assigned_spt - assigned_spt_float) < 0.0)[0]
        idx_equa = np.where((assigned_spt - assigned_spt_float) == 0.0)[0]

        # equal to M= Q x spt
        if assigned_spt.sum() > num_allocated_total_pt:
            delta_num = assigned_spt.sum() - num_allocated_total_pt
            selected_idx = np.asarray(list(idx_plus) + list(idx_equa) + list(idx_mius))
            assigned_spt[selected_idx[:delta_num]] += -1
        elif assigned_spt.sum() < num_allocated_total_pt:
            delta_num = num_allocated_total_pt - assigned_spt.sum()
            selected_idx = np.asarray(list(idx_mius) + list(idx_equa) + list(idx_plus))
            assigned_spt[selected_idx[:delta_num]] += 1
        else:
            pass

        return assigned_spt

    def get_batch_taus(self,X,num_data,num_sample,random_sample = True):
        if random_sample:        
            idx = np.random.choice(num_data, num_sample, replace=False)       
        else:
            idx = np.arange(self.call_num*num_sample,(self.call_num+1)*num_sample) % num_data        
        return self._get_subtauset(X[idx]),idx

    
    def calc_sptratio_given_X(self, weight_param, mu_param, std_param, X , intrain = True):
        if torch.is_tensor(weight_param):
            weight_param = weight_param.cpu().data.numpy()
        if torch.is_tensor(mu_param):
            mu_param = mu_param.cpu().data.numpy()
        if torch.is_tensor(std_param):
            std_param = std_param.cpu().data.numpy()

        num_data, dim = X.shape
        num_sample = int(num_data * self.rate)
        sub_sampled_tau,idx = self.get_batch_taus(X, num_data,num_sample,random_sample = False)        
        
        nominator_list = []
        for ith_weight, ith_mu, ith_std in zip(weight_param, mu_param, std_param):
            variance_sum = self.sum_g_tau(ith_mu, ith_std, inputs=sub_sampled_tau)
            covariance_sum = 0.0
            nominator_list.append(ith_weight * np.sqrt(variance_sum + covariance_sum))

        ratio = np.clip(nominator_list, a_min=lower_bound_weight, a_max=upper_bound_weight)
        ratio = softmax(np.log(ratio / self.temperature)).squeeze()

        # float to integers
        assigned_spt = self.float_to_integer(ratio)
        assigned_spt += self.num_min_pt
        
        if intrain == True:
            self.call_num +=1                    
        else:
            pass
        
        assert (assigned_spt.sum() == self.total_spt)
        return assigned_spt, ratio

    
    
    def calc_sptratio_given_XY(self, weight_param, mu_param, std_param, X, Y, alpha=None  , intrain = True):
        if torch.is_tensor(weight_param):
            weight_list = weight_param.cpu().data.numpy()
        if torch.is_tensor(mu_param):
            mu_list = mu_param.cpu().data.numpy()
        if torch.is_tensor(std_param):
            std_list = std_param.cpu().data.numpy()

        if alpha == []:
            ratio = np.ones(self.num_Q)/self.num_Q
        else:
            num_data, dim = X.shape
            num_sample = int(num_data * self.rate)
            sub_sampled_tau,idx = self.get_batch_taus(X, num_data,num_sample,random_sample = True)        
            alpha_idx = alpha[idx]
            
            out1 = K_SM_Components(mu_list, std_list, 2*sub_sampled_tau)
            out2 = -2 * K_SM_Components(mu_list, std_list, sub_sampled_tau) ** 2
            g_mat_list = np.ones_like(out1) + out1 + out2
            g_mat_list = np.sqrt(g_mat_list)

            ratio = []
            for ith_weight, ith_g_mat in zip(weight_list, g_mat_list):
                weighted_secondorder_term = (alpha_idx.T).dot(ith_g_mat.dot(alpha_idx)) + 1e-16
                ith_ratio = np.sqrt(  .5*np.clip(weighted_secondorder_term,a_min = 1e-16, a_max= None) )
                ratio.append(ith_weight * ith_ratio)

            ratio = np.asarray(ratio).squeeze()
            ratio = np.clip(ratio, a_min=lower_bound_weight, a_max=upper_bound_weight)
            ratio = softmax(np.log(ratio / self.temperature)).squeeze()
                    
        assigned_spt = self.float_to_integer(ratio)
        assigned_spt += self.num_min_pt
        
        if intrain == True:
            self.call_num +=1            
        else:
            pass
        assert (assigned_spt.sum() == self.total_spt)
        return assigned_spt, ratio






if __name__ == "__main__":
    ith_weight = [10., 0.5, 10., .5, 5]
    ith_mu = np.array([1, 5, 10, 20, 30]).reshape(-1, 1)
    ith_std = np.array([.0899, .0444, 0.0688, .0980, .0177]).reshape(-1, 1)

    # X = np.random.randn(10,3)
    X = np.arange(0, 10, 0.05).reshape(-1, 1)

    SMspt_manager = spt_manager(total_spt=500, num_Q=5)
    SMspt_manager.set_tau_collection(X, istau=True)

    assigned_spt_cov, ratio_cov = SMspt_manager.calc_sptratio(weight_param=ith_weight, mu_param=ith_mu,
                                                              std_param=ith_std)
    print('')
    assigned_spt, ratio = SMspt_manager.calc_sptratio(weight_param=ith_weight, mu_param=ith_mu, std_param=ith_std)
    print('')

    print('#' * 100)
    print('assigned_spt_cov,assigned_spt_cov.sum()')
    print(assigned_spt_cov, assigned_spt_cov.sum())

    print('assigned_spt,assigned_spt.sum()')
    print(assigned_spt, assigned_spt.sum())