from models_utility.param_gp import Param
from models_utility.spt_manager_train import spt_manager_train
from models_utility.function_gp import cholesky, lt_log_determinant
from torch import triangular_solve
from models_utility.likelihoods import Gaussian


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.set_default_tensor_type(torch.DoubleTensor)


zitter = 1e-8
pi2 = torch.tensor([2*np.pi])
#class ssgpr_rep_sm(gpmodel):
class ssgpr_rep_sm(nn.Module):
    
    def __init__(self, num_batch, num_sample_pt, param_dict, kernel=None, likelihood=None, device = None ):
        super(ssgpr_rep_sm, self).__init__()
        
        self.device = torch.device("cuda") if device else torch.device('cpu')
        self.name = None
        self._set_up_param(param_dict)
        self.num_batch = num_batch
        self.num_samplept = num_sample_pt
        self.total_num_sample = self.num_samplept*self.weight.numel()
        self.spt_manager = spt_manager_train(spt = num_sample_pt,  num_Q = self.num_Q ,rate= param_dict['weight_rate'])              
        self.likelihood = Gaussian(variance= self.noise, device = device) if likelihood == None else likelihood
        self.num_samplept_list_at = None
        #self.labmda_w = 0.0

        
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

    
    def _get_param(self):
        for ith in self.parameters():
            print('%s : %s'%(ith.param_name,ith.transform()))

            

    def _set_up_param(self, param_dict):
        self.input_dim = param_dict['input_dim']
        self.num_Q = param_dict['Num_Q']
        self.lr_hyp = param_dict['lr_hyp']
        self.inputs_dims = self.mu.shape[1]
        self.sampling_option = param_dict['sampling_option']
        #self.tau0 = param_dict['tau0']
        #self.tau = self.tau0
        self.noise = param_dict['noise_err']


        hypparam_dict = param_dict['hypparam']
        self.sf2 = Param(torch.tensor(hypparam_dict['noise_variance']).to(self.device),
                         requires_grad=False, requires_transform=True , param_name='sf2')

        self.weight = Param(torch.tensor(hypparam_dict['weight']).view(-1,1).to(self.device) ,
                            requires_grad=True, requires_transform=True , param_name='weight')
        self.mu = Param(torch.tensor(hypparam_dict['mean'] ).view(-1,self.input_dim).to(self.device),
                        requires_grad=True, requires_transform=True, param_name='mu')
        self.std = Param(torch.tensor(hypparam_dict['std']).view(-1, self.input_dim).to(self.device),
                         requires_grad=True,requires_transform=True, param_name='std')

        self.mu_prior = Param(torch.tensor(hypparam_dict['mean']).view(-1,self.input_dim).to(self.device) ,
                              requires_grad=False, requires_transform=True , param_name='mu_prior')
        self.std_prior = Param(torch.tensor(hypparam_dict['std']).view(-1,self.input_dim).to(self.device),
                               requires_grad=False, requires_transform=True , param_name='std_prior')

        return


    def _set_data(self,x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.spt_manager._set_collection_tauset(x_train)
        return 
    
    


    def _set_num_spectralpt(self,num_pt):
        self.num_samplept = num_pt
        self.total_num_sample = self.num_samplept*self.weight.numel()
        self.spt_manager._set_num_spectralpt(num_pt) 
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
            assign_rate = F.softmax(self.weight,dim = 0).squeeze()
            assigned_spt = [max(int(ith),1) for ith in self.total_num_sample * assign_rate]
            outs = assigned_spt
        else:
            outs = [self.num_samplept for ith in range(self.num_Q)]

        self.num_samplept_list_at = outs    
        return outs
    
    
    def _sampling_gaussian(self, mu, std, num_sample):
        eps = Variable(torch.randn(num_sample, self.input_dim).to(self.device))
        return mu + std.mul(eps)



    def _compute_gaussian_basis(self, x, xstar=None):
        sampled_spectral_pt = self._sampling_gaussian(self.mu.transform(),self.std.transform(),self.num_sample_pt)  # self.num_sample x dim
        xdotspectral = x.matmul(sampled_spectral_pt.t())
        Phi = torch.cat([xdotspectral.cos(), xdotspectral.sin()], 1).to(self.device)
        if xstar is None:
            return Phi
        else:
            xstardotspectral = xstar.matmul(sampled_spectral_pt.t())
            Phi_star = torch.cat([xstardotspectral.cos(), xstardotspectral.sin()], 1).to(self.device)
            return Phi, Phi_star



    def _compute_sm_basis(self, x , xstar=None, intrain = True):
        multiple_Phi = []
        current_sampled_spectral_list = []
        if self.weight.shape[0] > 1:
            current_pi = self.weight.transform().reshape([1, -1]).squeeze()
        else:
            current_pi = self.weight.transform()

        self._assign_num_spectralpt(x,intrain)                       
        num_samplept_list = self.num_samplept_list_at 

        for i_th in range(self.weight.numel()):
            ith_allocated_sample = num_samplept_list[i_th]
            sampled_spectal_pt = self._sampling_gaussian(mu = self.mu.transform()[i_th],std = self.std.transform()[i_th],num_sample = ith_allocated_sample)  # self.num_sample x dim

            if xstar is not None:
                current_sampled_spectral_list.append(sampled_spectal_pt)
            xdotspectral = (2 * np.pi) * x.matmul(sampled_spectal_pt.t())

            Phi_i_th = (current_pi[i_th] / ith_allocated_sample).sqrt() * torch.cat([xdotspectral.cos(), xdotspectral.sin()], 1).to(self.device)
            multiple_Phi.append(Phi_i_th)

        if xstar is None:
            return torch.cat(multiple_Phi, 1)
        else:
            multiple_Phi_star = []
            for i_th, current_sampled in enumerate(current_sampled_spectral_list):
                xstardotspectral = (2 * np.pi) * xstar.matmul(current_sampled.t())
                Phistar_i_th = (current_pi[i_th] / len(current_sampled)).sqrt() * torch.cat([xstardotspectral.cos(), xstardotspectral.sin()],1).to(self.device)
                multiple_Phi_star.append(Phistar_i_th)
            return torch.cat(multiple_Phi, 1), torch.cat(multiple_Phi_star, 1)


    def _compute_gram_approximate(self, Phi):
        return  Phi.t().matmul(Phi) + (self.likelihood.variance.transform()**2 + zitter).expand(Phi.shape[1], Phi.shape[1]).diag().diag()



    def _compute_kernel_sm_approximate(self, x , normalized_option = False, usepairwise = False):
        if usepairwise is False:
            Phi_list = self._compute_sm_basis(x)
        else:
            Phi_list = self._compute_sm_basis(x, usepairwise = usepairwise)


        kernel_output =  Phi_list.matmul(Phi_list.t())
        if normalized_option == True:
            return kernel_output/kernel_output[0,0]
        else:
            return kernel_output




    def compute_loss(self,batch_x,batch_y,kl_option,current_iter = 1):
        """
        :param batch_x:
        :param batch_y:
        :return: approximate lower bound of negative log marginal likelihood
        """

        
        num_input = batch_x.shape[0]
        loss = 0
        for j_th in range(self.num_batch):
            Phi = self._compute_sm_basis(batch_x,intrain = True)
            Approximate_gram = self._compute_gram_approximate(Phi)
            L = cholesky(Approximate_gram)
            Lt_inv_Phi_y = trtrs(L, Phi.t(), lower=True).matmul(batch_y)  
            
            loss += (0.5 / self.likelihood.variance.transform()**2) * (batch_y.pow(2).sum() - Lt_inv_Phi_y.pow(2).sum())
            loss += lt_log_determinant(L)
            loss += (-self.total_num_sample)*2*self.likelihood.variance
            loss += 0.5 * num_input * (np.log(2*np.pi) + 2*self.likelihood.variance )


        return (1 / self.num_batch) * loss 



    def _predict(self, inputs_new, diag = True):
        if isinstance(inputs_new, np.ndarray):
            inputs_new = Variable(torch.Tensor(inputs_new).to(self.device), requires_grad=False)

        #self._assign_num_spectralpt(self.x,usepairwise=None)            
        Phi, Phi_star = self._compute_sm_basis(self.x, inputs_new,intrain = False)
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


        
    def _predict_exact_batch(self, inputs_new, diag=True):
        batch_num = 1000
        n = len(inputs_new)//batch_num
        mean_f = []
        var_f = []
        for i in range(1,n+1):
            idx = np.arange( (i-1)*batch_num,i*batch_num)
            batch_mean_f,batch_var_f = self._predict_exact(inputs_new[idx],diag= True)
            mean_f.append(batch_mean_f)
            var_f.append(batch_var_f)

        batch_mean_f,batch_var_f = self._predict_exact(inputs_new[n*batch_num:],diag= True)
        mean_f.append(batch_mean_f)
        var_f.append(batch_var_f)


        return torch.cat(mean_f,dim=0),torch.cat(var_f,dim=0)



if __name__ == "__main__":
    #1d inputs
    from utility.dataset import _load_collection_data
    from models_utility.construct_models import _initialize_SMkernel_hyp
    import matplotlib.pyplot as plt

    device = True
    path_filename = 'CO2'
    x_train,x_test,y_train,y_test = _load_collection_data(path_filename,cuda_option=device)

    random_seed= 1000
    setting_dict = {}
    setting_dict['random_seed'] = random_seed
    setting_dict['Num_Q'] = 10
    setting_dict['input_dim'] = 1
    setting_dict['lr_hyp'] = 0.005
    setting_dict['sampling_option'] = 'weight'
    setting_dict['tau0'] = 1000

    param_dict = _initialize_SMkernel_hyp(x_train, y_train, setting_dict, random_seed)
    num_sample_pt = 5
    num_batch = 1

    mu = torch.cat([y_train,y_test],dim = 0).mean()
    y_train -= mu
    y_test -=mu


    model = ssgpr_rep_sm(num_batch = num_batch,
                         num_sample_pt = num_sample_pt,
                         param_dict=param_dict,
                         device = device)


    print(model.tau)


    for i in range(20000+1):
        model.train()
        #loss = model.compute_loss(batch_x=x_train, batch_y=y_train, kl_option=True , current_iter = 1+i)
        loss = model.compute_loss(batch_x=x_train, batch_y=y_train, current_iter=1 + i)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

        if i %2500 == 0:
            with torch.no_grad():
                mean_f, mean_var = model._predict(inputs_new = x_test)
                print('%d th loss : %.4f, test mse : %.4f'%(i,
                                                            loss.cpu().data.numpy(),
                                                            (mean_f-y_test).pow(2).sum().cpu().data.numpy()))
                print('model._assign_num_spectralpt()')
                print(model._assign_num_spectralpt())
                print(model.tau)

                plt.figure(figsize=(10,5))
                plt.plot(x_test.cpu().data.numpy(),mean_f.cpu().data.numpy(),'b')
                plt.plot(x_test.cpu().data.numpy(),y_test.cpu().data.numpy(),'r')
                plt.title('current iter :%d'%(i))
                plt.show()
