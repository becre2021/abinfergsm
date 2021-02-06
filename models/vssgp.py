import numpy as np
np.set_printoptions(precision = 4)

import torch
import math
import torch.nn as nn
from models_utility.param_gp import Param
from models_utility.function_gp import cholesky, lt_log_determinant
from models_utility.likelihoods import Gaussian


pi = math.pi
class vssgp(nn.Module):
    def __init__(self, train_X, train_Y, setting_dict, device):

        # N, D
        # Q : The number of component SM
        # K : The number of spectral poinst for each component
        # M=QK : The total number of spectral points for each componet

        super(vssgp, self).__init__()

        self.device = torch.device("cuda") if device else torch.device('cpu')
        self.name = 'VSSGP'
        self.Q = setting_dict['Num_Q']
        self.D = setting_dict['input_dim']
        self.D_Y = 1  # IF multiouput, then D_Y > 1
        self.K = setting_dict['num_sample_pt']
        self.total_sample_pt = self.Q * self.K
        self._set_data(batch_x=train_X, batch_y=train_Y)
        #self.likelihood = Gaussian(variance= [setting_dict['noise_err']], device = device) 


        # inverse of noise error
        tau = 1 / (1e-16 + setting_dict['noise_err'])
        self.tau = Param(torch.tensor([tau]).to(self.device), requires_grad=True, requires_transform=True,param_name='tau')  # slope
        #self.tau  = 1 / (1e-16 + self.likelihood.variance.transform())

        # hyperparameter
        hypparam_dict = setting_dict['hypparam']
        self.weight = Param(torch.tensor(hypparam_dict['weight'].transpose()[0]).to(self.device), requires_grad=True,
                            requires_transform=True, param_name='weight')
        self.std = Param(torch.tensor(hypparam_dict['std'].transpose()).to(self.device), requires_grad=True,
                         requires_transform=True,param_name='std')  # D x Q   transpose needed for our initialization Q x D
        self.mu = Param(torch.tensor(hypparam_dict['mean'].transpose()).to(self.device), requires_grad=True,
                        requires_transform=True,param_name='mu')  # D x Q   transpose needed for our initialization Q x D

        
        alpha, alpha_delta = (pi * torch.rand(self.K, self.Q)), (pi * torch.rand(self.K, self.Q))
        alpha = alpha % (2 * pi)
        beta = (alpha + alpha_delta) % (2 * pi)
        self.alpha = Param(alpha.to(self.device), requires_grad=True, requires_transform=True,param_name='alpha')  # cos(2pi s^T x + b_k)  b_k ~ \alpha,\beta
        self.beta = Param(beta.to(self.device), requires_grad=True, requires_transform=True,param_name='beta')  # cos(2pi s^T x + b_k)  b_k ~ \alpha,\beta

        # variational spectral points parameter for p(W)
        #var_mu = 5 * torch.randn(self.D, self.K, self.Q)  #syn
        var_mu = .5 * torch.rand(self.D, self.K, self.Q)  #regression
        var_sigma = 0.05 * torch.rand(self.D, self.K, self.Q)
        self.var_mu_w = Param(var_mu.to(self.device), requires_grad=True, requires_transform=False,
                              param_name='var_mu_for_alpha')  # (K x Q)  x D_Y
        self.var_sigma_w = Param(var_sigma.to(self.device), requires_grad=True, requires_transform=True,
                                 param_name='var_std_for_alpha')  # (K x Q) x D_Y


        #variational weights parameter for p(A)        
        mu0 = torch.randn((self.K * self.Q), self.D_Y)                    
        sigma0 = .1 * torch.rand((self.K * self.Q), self.D_Y)
        self.var_mu_a = Param(mu0.to(self.device), requires_grad=True, requires_transform=False, param_name='prior_mu')  # D x K x Q
        self.var_sigma_a = Param(sigma0.to(self.device), requires_grad=True, requires_transform=True,param_name='prior_std')  # D x K x Q

        #slop parameters
        a = 0.001*torch.randn(self.D, 1)
        self.a = Param(torch.tensor(a).to(self.device), requires_grad=False, requires_transform=False,
                       param_name='meanfun_slope')  # slope
        self.b = Param(torch.tensor([0]).to(self.device), requires_grad=False, requires_transform=False,
                       param_name='meanfun_intercept')  # intersept




    def _set_data(self, batch_x, batch_y):
        if batch_x is None:
            self.X = None
            self.Y = None
        else:
            if torch.is_tensor(batch_x):
                self.X = batch_x  # N x D
                self.Y = batch_y  # N x 1
            else:
                self.X = torch.from_numpy(batch_x).to(self.device)
                self.Y = torch.from_numpy(batch_y).to(self.device)


    def _set_inducing_pt(self,num_pt):
        #num_pt : self.K x self.Q
        idx = np.random.choice(len(self.X),num_pt)        
        #z = self.X[idx].t()[:, None, None] + 0.1*torch.randn(self.D, self.K, self.Q).to(self.device)
        z = self.X[idx].reshape(self.D, self.K, self.Q) + 0.1*torch.randn(self.D, self.K, self.Q).to(self.device)
        self.z = Param(z.to(self.device), requires_grad=True, requires_transform=False,param_name='inducing_spectral_pt')  # inducing points z_k    D x K x Q

        return 
    
            
    def get_Ephi(self, Xstar=None):
        if Xstar is None:
            N, D = self.X.shape
            X = self.X
        else:
            N, D = Xstar.shape
            X = Xstar


        two_over_K = (2. * self.weight.transform()[None, None, :]) / self.K
        mean_p = 1 / (self.mu.transform() + 1e-8)
        std_p = 1 / (2 * pi * self.std.transform() + 1e-8)
        E_w = mean_p[:, None, :] + std_p[:, None, :] * (self.var_mu_w.transform())
        xbar = 2 * pi * (X[:, :, None, None] - self.z[None, :, :, :])

        cos_w = torch.cos(self.alpha.transform() + (xbar * E_w[None, :, :, :]).sum(dim=1))
        exp_decay_term = torch.exp(-0.5 * ((std_p[None, :, None, :] * xbar).pow(2) * self.var_sigma_w.transform()[None, :, :, :]).sum(dim=1))

        E_phi = ((two_over_K ** 0.5) * exp_decay_term * cos_w).view(N, -1)  # N x DQ

        cos_2w = torch.cos(2 * self.alpha.transform() + 2 * (xbar * E_w[None, :, :, :]).sum(dim=1))
        E_cos_sq = two_over_K * (0.5 + (0.5 * exp_decay_term.pow(4)) * cos_2w)

        E_phiT_E_phi = torch.mm(E_phi.t(), E_phi)
        E_phiT_E_phi += -E_phiT_E_phi.diag().diag() + E_cos_sq.sum(dim=0).view(-1).diag()
        return E_phi, E_phiT_E_phi, E_cos_sq


    
    def get_opt_A(self, EphiTphi=None, yT_Ephi=None):
        if EphiTphi is None and yT_Ephi is None:
            Ephi, EphiTphi, E_cos_sq = self.get_Ephi()
            yT_Ephi = (self.Y.t()).mm(Ephi)

        SigInv = EphiTphi + ( (1 / self.tau.transform()**2) + 1e-8) * torch.eye(EphiTphi.shape[0]).to(self.device)
        cholTauSigInv = (self.tau.transform()) * cholesky(SigInv)
        invCholTauSigInv = torch.cholesky_inverse(cholTauSigInv)
        tauInvSig = invCholTauSigInv.t().mm(invCholTauSigInv)
        Sig_EPhiT_Y = (self.tau.transform()**2) * tauInvSig.mm(yT_Ephi.t())
        return Sig_EPhiT_Y, tauInvSig, cholTauSigInv


    def kl_gaussian(self, var_mu, var_sigma):
        # between q(s;mu,sigma)||N(s;0,1)
        return 0.5 * (var_sigma.transform() + (var_mu.transform() ** 2) - var_sigma - 1.0).sum()



    def compute_loss(self,batch_x, batch_y, kl_option ,use_opt_A=False):
        # LL : log likelihood term
        # kl : Kl term
        # -(LL-KL) is a cost to be minimized
        N, D = self.Y.shape
        y = self.Y - (self.X.mm(self.a) + self.b)
        Ephi, EphiT_Ephi, E_cos_sq = self.get_Ephi()
        yT_Ephi = torch.mm(y.t(), Ephi)

        if use_opt_A is True:
            opt_A_mean, opt_A_cov, cholSigInv = self.get_opt_A()
            LL = -0.5 * N * D * np.log(2 * pi) 
            LL += 0.5 * N * D * (2*self.tau)
            LL += -0.5 * (self.tau.transform()**2) * (y.pow(2)).sum(dim=0).squeeze()
            LL += 0.5 * (self.tau.transform()**2) * (opt_A_mean.t() * yT_Ephi).sum()
            LL += -0.5 * D * (2 * torch.log(cholSigInv.diag())).sum()
            kl_w = self.kl_gaussian(self.var_mu_w, self.var_sigma_w)
            return -(LL - kl_w)


        else:
            LL = -0.5 * N * D * np.log(2 * pi) 
            LL += 0.5 * N * D * (2*self.tau)
            LL += -0.5 *(self.tau.transform()**2) * (y.pow(2)).sum(dim=0).squeeze()

            vmvmT = (self.var_mu_a.transform()[:, None, :] * self.var_mu_a.transform()[None, :, :]).sum(dim=2)
            vsigmaDiag = (self.var_sigma_a.transform()).sum(dim=1).diag()
            LL += -0.5 * (self.tau.transform()**2) * torch.sum(EphiT_Ephi * (vsigmaDiag + vmvmT))
            
            
            YTEhi = (torch.mm(self.Y.t(), Ephi))  # 1 x (Q X K)            x
            sum_YTEhi_vmT = (YTEhi * (self.var_mu_a.transform()).t()).sum()
            LL += (self.tau.transform()**2) * sum_YTEhi_vmT


            kl_w = self.kl_gaussian(self.var_mu_w, self.var_sigma_w)
            kl_a = self.kl_gaussian(self.var_mu_a, self.var_sigma_a)
            KL = kl_w + kl_a

            return -(LL - KL)

    # variation
    def _predict(self, inputs_new , diag=True, use_opt_A=False):
        N, _ = inputs_new.shape
        Ephistar, _, E_cos_sqstar = self.get_Ephi(Xstar=inputs_new)
        if use_opt_A is True:
            opt_A_mean, opt_A_cov, cholSigInv = self.get_opt_A()
            y_pred_mean = Ephistar.mm(opt_A_mean) + (inputs_new.mm(self.a.transform()) + self.b.transform())

            EphiTphi = Ephistar[:, :, None] * Ephistar[:, None, :]  # N x K*comp x K*comp
            EphiTphi += -torch.eye(self.K * self.Q).to(self.device)[None, :, :] * EphiTphi + torch.eye(self.K * self.Q).to(self.device)[None, :, :] * E_cos_sqstar.view(N, -1)[:, :, None]

            Psi = (torch.sum(EphiTphi * opt_A_cov[None, :, :], dim=2)).sum(dim=1)
            Psi = Psi.view(-1, 1)
            Delta_E = EphiTphi - Ephistar[:, :, None] * Ephistar[:, None, :]
            mu_a = (self.var_mu_a.transform()).mm(self.var_mu_a.transform().t())

            y_pred_var = Psi
            y_pred_var += (Delta_E.view(N, -1) * mu_a.view(1, -1)).sum(dim=1, keepdim=True)
            y_pred_var += 1 / (self.tau.transform()**2)
            return y_pred_mean, y_pred_var

        else:
            y_pred_mean = Ephistar.mm(self.var_mu_a.transform()) + (inputs_new.mm(self.a.transform()) + self.b.transform())
            Psi = (E_cos_sqstar.flatten(1)[:,:,None]*self.var_sigma_a.transform()[None, :, :]).squeeze()
            flag_diag_n = E_cos_sqstar.flatten(1) - Ephistar**2

            M = self.var_mu_a.transform()
            y_pred_var = ((M.transform().t()).matmul(flag_diag_n[:,:,None]*M)).squeeze(dim=2)
            y_pred_var += (Psi[:, :, None]**2).sum(dim=1)
            y_pred_var += (1/(self.tau.transform()**2+1e-16)).squeeze()
            return y_pred_mean, y_pred_var
        
        

 
