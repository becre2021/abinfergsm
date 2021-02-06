import torch
import random
import numpy as np
import math
from itertools import combinations
from scipy.special import softmax

class spt_manager(object):
    def __init__(self, total_spt, num_Q, rate = 0.01):
        self.tau = 1
        #self.tau = 10        
        self.spt = int(total_spt/num_Q)
        self.total_spt = total_spt
        self.num_Q = num_Q
        self.tau_collection = None
        self.ij_taus_sum = None
        self.ij_taus_minus = None
        self.i_taus = None
        self.j_taus = None
        self.rate = rate
        self.num_tau_limit = 1000000


    def k_sm(self, ith_weight, ith_mu, ith_std, tau_collection):
        exp_term_in = ((tau_collection * ith_std) ** 2).sum(axis=1, keepdims=True)
        exp_term = np.exp(-2 * (math.pi ** 2) * exp_term_in)
        cos_term_in = (tau_collection * ith_mu).sum(axis=1, keepdims=True)
        cos_term = np.cos(2 * math.pi * cos_term_in)
        return ith_weight * (exp_term * cos_term)


    def set_tau_collection(self, X, istau=False, usepairwise = None):
        if self.tau_collection is None:
            if torch.is_tensor(X):
                X = X.cpu().data.numpy()

            if istau:
                self.tau_collection = X
            else:
                self.tau_collection = []
                for ith,ith_component in enumerate(combinations(X, 2)):
                    self.tau_collection.append(ith_component[0] - ith_component[1])
                    if ith > self.num_tau_limit:
                        break
                self.tau_collection = np.asarray(self.tau_collection)
            self.num_tau = len(self.tau_collection)

            print('spt manager num tau : %d'%(self.num_tau))


            if usepairwise is not None:
                self.pairwise_tau_collection = combinations(self.tau_collection,2)
                ij_taus_sum, ij_taus_minus, i_taus, j_taus = [], [], [], []
                for ith_tau, jth_tau in combinations(self.tau_collection, 2):
                    ij_taus_sum.append(ith_tau + jth_tau)
                    ij_taus_minus.append(ith_tau - jth_tau)
                    i_taus.append(ith_tau)
                    j_taus.append(jth_tau)

                self.ij_taus_sum = np.asarray(ij_taus_sum)
                self.ij_taus_minus = np.asarray(ij_taus_minus)
                self.i_taus = np.asarray(i_taus)
                self.j_taus = np.asarray(j_taus)
        else:
            pass

        return



    def g_tau(self, mu, std ,inputs):
        out = 1 + self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection= 2*inputs)
        out += -2 * (self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=inputs)) ** 2
        return out


    def sum_g_tau(self, mu, std ,inputs):
        return (self.g_tau(mu, std ,inputs )).sum(axis=0)



    def h_taupair(self, mu, std, i_taus, j_taus, ij_taus_sum, ij_taus_minus):
        out = -self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=i_taus) * self.k_sm(ith_weight=1.0,
                                                                                                         ith_mu=mu,
                                                                                                         ith_std=std,
                                                                                                         tau_collection=j_taus)
        out += 0.5 * self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=ij_taus_sum)
        out += 0.5 * self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=ij_taus_minus)
        return out


    def sum_h_taupair(self, mu, std ,i_taus, j_taus, ij_taus_sum, ij_taus_minus):
        return (self.h_taupair(mu, std,i_taus, j_taus, ij_taus_sum, ij_taus_minus )).sum(axis=0)



    def calc_sptratio(self, weight_param, mu_param, std_param):
        if torch.is_tensor(weight_param):
            weight_param = weight_param.cpu().data.numpy()
        if torch.is_tensor(mu_param):
            mu_param = mu_param.cpu().data.numpy()
        if torch.is_tensor(std_param):
            std_param = std_param.cpu().data.numpy()

        if self.num_tau <= self.num_tau_limit:
            if self.rate == 1.0:
                sample_idx = np.arange(self.num_tau)
            else:
                sample_idx = np.random.choice(np.arange(self.num_tau),size = int(self.rate*self.num_tau))
        else:
            sample_idx = np.random.choice(np.arange(self.num_tau_limit),size = int(self.rate*self.num_tau))


        nominator_list = []
        for ith_weight, ith_mu, ith_std in zip(weight_param, mu_param, std_param):
            # variance_term
            variance_sum = .5 * self.sum_g_tau(ith_mu, ith_std, inputs=self.tau_collection[sample_idx])
            # covariance_term
            if self.ij_taus_sum is not None:
                covariance_sum = 2 * self.sum_h_taupair(ith_mu, ith_std, i_taus = self.i_taus, j_taus = self.j_taus, ij_taus_sum = self.ij_taus_sum ,ij_taus_minus = self.ij_taus_minus)
            else:
                covariance_sum = [0]


            # nominator
            nominator_list.append(ith_weight * np.sqrt(variance_sum + covariance_sum))

        ratio = np.asarray(nominator_list) / (np.asarray(nominator_list).sum() + 1e-8)
        ratio = np.clip(ratio/self.tau, a_min=1e-8,a_max = None)                     
        ratio = softmax(np.log( ratio ))


        
#         ####################
#         # original training
#         assigned_spt = np.asarray([int(np.round(ipt))  for ipt in (self.total_spt)*ratio])
#         # minimum condition
#         for ith in range(len(assigned_spt)) :
#             if assigned_spt[ith] <= 1:
#                 assigned_spt[ith] = 2    
                
#         max_idx = np.argmax(assigned_spt)
#         #equal to M= Q x spt
#         if assigned_spt.sum() > self.total_spt:
#             delta_num = assigned_spt.sum() - self.total_spt
#             selected_idx = np.argsort(-assigned_spt)[:delta_num] #largest sorting
#             #selected_idx = np.argsort(-assigned_spt)[:delta_num] #smallest sorting       
#             assigned_spt[selected_idx] -= 1
            
#         elif assigned_spt.sum() < self.total_spt:
#             delta_num = self.total_spt - assigned_spt.sum()            
#             selected_idx = np.argsort(-assigned_spt)[:delta_num] #largest sorting
#             #selected_idx = np.argsort(assigned_spt)[:delta_num] #smallest sorting            
#             assigned_spt[selected_idx] += 1            
#         else:
#             pass



        ###################
        #robust training
        assigned_spt = np.asarray([int(np.round(ipt)) + 1 for ipt in (int(self.total_spt/2))*ratio])
        assigned_spt += int(self.total_spt/(2*self.num_Q))
        max_idx = np.argmax(assigned_spt)

        #equal to M= Q x spt
        if assigned_spt.sum() > self.total_spt:
            delta_num = assigned_spt.sum() - self.total_spt
            selected_idx = np.argsort(-assigned_spt)[:delta_num] #largest sorting
            #selected_idx = np.argsort(-assigned_spt)[:delta_num] #smallest sorting       
            assigned_spt[selected_idx] -= 1
            
        elif assigned_spt.sum() < self.total_spt:
            delta_num = self.total_spt - assigned_spt.sum()            
            #selected_idx = np.argsort(-assigned_spt)[:delta_num] #largest sorting
            selected_idx = np.argsort(assigned_spt)[:delta_num] #smallest sorting            
            assigned_spt[selected_idx] += 1            
        else:
            pass
        

#         print('assigned_spt,ratio')
#         print(assigned_spt,ratio)

        
        return assigned_spt, ratio




# class spt_manager(object):
#     def __init__(self, total_spt, num_Q, rate = 0.01):
#         self.tau = 10
#         self.total_spt = total_spt
#         self.num_Q = num_Q
#         self.tau_collection = None
#         self.ij_taus_sum = None
#         self.ij_taus_minus = None
#         self.i_taus = None
#         self.j_taus = None
#         self.rate = rate
#         self.num_tau_limit = 3000


#     def k_sm(self, ith_weight, ith_mu, ith_std, tau_collection):
#         exp_term_in = ((tau_collection * ith_std) ** 2).sum(axis=1, keepdims=True)
#         exp_term = np.exp(-2 * (math.pi ** 2) * exp_term_in)
#         cos_term_in = (tau_collection * ith_mu).sum(axis=1, keepdims=True)
#         cos_term = np.cos(2 * math.pi * cos_term_in)
#         return ith_weight * (exp_term * cos_term)


#     def set_tau_collection(self, X, istau=False, usepairwise = None):
#         if self.tau_collection is None:
#             if torch.is_tensor(X):
#                 X = X.cpu().data.numpy()

#             if istau:
#                 self.tau_collection = X
#             else:
#                 self.tau_collection = []
#                 for ith,ith_component in enumerate(combinations(X, 2)):
#                     self.tau_collection.append(ith_component[0] - ith_component[1])
#                     if ith > self.num_tau_limit:
#                         break
#                 self.tau_collection = np.asarray(self.tau_collection)
#             self.num_tau = len(self.tau_collection)

#             print('spt manager num tau : %d'%(self.num_tau))


#             if usepairwise is not None:
#                 self.pairwise_tau_collection = combinations(self.tau_collection,2)
#                 ij_taus_sum, ij_taus_minus, i_taus, j_taus = [], [], [], []
#                 for ith_tau, jth_tau in combinations(self.tau_collection, 2):
#                     ij_taus_sum.append(ith_tau + jth_tau)
#                     ij_taus_minus.append(ith_tau - jth_tau)
#                     i_taus.append(ith_tau)
#                     j_taus.append(jth_tau)

#                 self.ij_taus_sum = np.asarray(ij_taus_sum)
#                 self.ij_taus_minus = np.asarray(ij_taus_minus)
#                 self.i_taus = np.asarray(i_taus)
#                 self.j_taus = np.asarray(j_taus)
#         else:
#             pass

#         return



#     def g_tau(self, mu, std ,inputs):
#         out = 1 + self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection= 2*inputs)
#         out += -2 * (self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=inputs)) ** 2
#         return out


#     def sum_g_tau(self, mu, std ,inputs):
#         return (self.g_tau(mu, std ,inputs )).sum(axis=0)



#     def h_taupair(self, mu, std, i_taus, j_taus, ij_taus_sum, ij_taus_minus):
#         out = -self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=i_taus) * self.k_sm(ith_weight=1.0,
#                                                                                                          ith_mu=mu,
#                                                                                                          ith_std=std,
#                                                                                                          tau_collection=j_taus)
#         out += 0.5 * self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=ij_taus_sum)
#         out += 0.5 * self.k_sm(ith_weight=1.0, ith_mu=mu, ith_std=std, tau_collection=ij_taus_minus)
#         return out


#     def sum_h_taupair(self, mu, std ,i_taus, j_taus, ij_taus_sum, ij_taus_minus):
#         return (self.h_taupair(mu, std,i_taus, j_taus, ij_taus_sum, ij_taus_minus )).sum(axis=0)



#     def calc_sptratio(self, weight_param, mu_param, std_param):
#         if torch.is_tensor(weight_param):
#             weight_param = weight_param.cpu().data.numpy()
#         if torch.is_tensor(mu_param):
#             mu_param = mu_param.cpu().data.numpy()
#         if torch.is_tensor(std_param):
#             std_param = std_param.cpu().data.numpy()

#         if self.num_tau <= self.num_tau_limit:
#             if self.rate == 1.0:
#                 sample_idx = np.arange(self.num_tau)
#             else:
#                 sample_idx = np.random.choice(np.arange(self.num_tau),size = int(self.rate*self.num_tau))
#         else:
#             sample_idx = np.random.choice(np.arange(self.num_tau_limit),size = int(self.rate*self.num_tau))


#         nominator_list = []
#         for ith_weight, ith_mu, ith_std in zip(weight_param, mu_param, std_param):
#             # variance_term
#             variance_sum = .5 * self.sum_g_tau(ith_mu, ith_std, inputs=self.tau_collection[sample_idx])
#             # covariance_term
#             if self.ij_taus_sum is not None:
#                 covariance_sum = 2 * self.sum_h_taupair(ith_mu, ith_std, i_taus = self.i_taus, j_taus = self.j_taus, ij_taus_sum = self.ij_taus_sum ,ij_taus_minus = self.ij_taus_minus)
#             else:
#                 covariance_sum = [0]


#             # nominator
#             nominator_list.append(ith_weight * np.sqrt(variance_sum + covariance_sum))

#         ratio = np.asarray(nominator_list) / (np.asarray(nominator_list).sum() + 1e-8)
#         ratio = np.clip(ratio/self.tau, a_min=1e-12,a_max = None)     
# #a = np.clip(a, 1e-12, 8)        
# #         print('ratio,self.tau')        
# #         print(ratio,self.tau)
                
#         ratio = softmax(np.log( ratio + 1e-08))


        
        
#         assigned_spt = np.asarray([int(np.round(ipt)) + 1 for ipt in (self.total_spt - self.num_Q) * ratio])
#         max_idx = np.argmax(assigned_spt)
        
#         #equal to M= Q x spt
#         if assigned_spt.sum() > self.total_spt:
#             delta_num = assigned_spt.sum() - self.total_spt
#             assigned_spt[max_idx] -= delta_num

#         elif np.asarray(assigned_spt).sum() < self.total_spt:
#             delta_num = self.total_spt - assigned_spt.sum()
#             assigned_spt[max_idx] += delta_num
#         else:
#             pass

        
# #         #equal to M= Q x spt
# #         if assigned_spt.sum() > self.total_spt:
# #             assigned_spt = np.asarray(assigned_spt)
# #             selected_idx = np.where(assigned_spt >= 2)[0]
# #             random.shuffle(selected_idx)
# #             delta_num = assigned_spt.sum() - self.total_spt
# #             for jth in selected_idx[:delta_num] :
# #                 assigned_spt[jth] -= 1

# #         elif np.asarray(assigned_spt).sum() < self.total_spt:
# #             assigned_spt = np.asarray(assigned_spt)
# #             selected_idx = np.where(assigned_spt >= 0)[0]
# #             random.shuffle(selected_idx)
# #             delta_num = self.total_spt - assigned_spt.sum()
# #             for jth in selected_idx[:delta_num] :
# #                 assigned_spt[jth] += 1
# #         else:
# #             pass



#         return assigned_spt, ratio



if __name__ == "__main__":
    ith_weight = [10., 0.5, 10., .5, 5]
    ith_mu = np.array([1, 5, 10, 20, 30]).reshape(-1, 1)
    ith_std = np.array([.0899, .0444, 0.0688, .0980, .0177]).reshape(-1, 1)


    # X = np.random.randn(10,3)
    X = np.arange(0, 10, 0.05).reshape(-1, 1)


    SMspt_manager = spt_manager(total_spt=500, num_Q=5)
    SMspt_manager.set_tau_collection(X, istau=True)

    assigned_spt_cov,ratio_cov = SMspt_manager.calc_sptratio(weight_param=ith_weight,mu_param=ith_mu,std_param=ith_std )
    print('')
    assigned_spt,ratio = SMspt_manager.calc_sptratio(weight_param=ith_weight,mu_param=ith_mu,std_param=ith_std )
    print('')

    print('#'*100)
    print('assigned_spt_cov,assigned_spt_cov.sum()')
    print(assigned_spt_cov,assigned_spt_cov.sum())

    print('assigned_spt,assigned_spt.sum()')
    print(assigned_spt,assigned_spt.sum())