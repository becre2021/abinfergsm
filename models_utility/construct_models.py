from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from kernels.RBF_kernel import RBF
from kernels.SM_kernel import SM
from models.gp import gpr
from models.gp_vfe import gp_vfe
from models.vssgp import vssgp
from models.gp_rff import ssgpr_sm
from models.gp_rrff_reg import ssgpr_rep_sm_reg
from models_utility.personalized_adam import Adam_variation
import torch


def _intialize_RBFkernel_hyp(x_train,y_train, setting_dict, random_seed):
    RBFKernel_hyp = {}
    RBFKernel_hyp['variance'] = 1.0  # real v2
    RBFKernel_hyp['length_scale'] = 1.0

    print('RBFKernel_hyp[length_scale]')
    print(RBFKernel_hyp['length_scale'])

    setting_dict['hypparam'] = RBFKernel_hyp

    return setting_dict


def _inverse_sampling_given_pdf(energies,empirical_pdf,sample_num):
    cum_prob = np.cumsum(empirical_pdf)
    R = np.random.uniform(0, 1, sample_num)
    gen_energies = [float(energies[np.argwhere(cum_prob == min(cum_prob[(cum_prob - r) > 0]))]) for r in R]
    return np.asarray(gen_energies).reshape(1,-1)




# single_inputs
def _initialize_SMkernelhyp( x_train,y_train, setting_dict, random_seed, yesSM = True, filename = None):
    """
    :param y_train:
    :param setting_dict: num_Q,input_dim
    :param random_seed:
    :return:
    """
    if yesSM is False:
        return _intialize_RBFkernel_hyp(x_train, y_train, setting_dict, random_seed)
    else:
        y_train = y_train.cpu().data.numpy()
        if int(1/(x_train[1]-x_train[0])) == 0:
            Fs = int(1/(x_train[1]-x_train[0]))+1
        else:
            Fs = int(1/(x_train[1]-x_train[0]))
            
        np.random.seed(random_seed)
        thres_hold = 0.0
        freqs, psd = signal.welch(y_train.reshape(1, -1).squeeze(), fs=Fs, nperseg=len(y_train))
        psd = psd/psd.sum(0)
        Num_Q = setting_dict['num_Q']


        SMKernel_hyp = {}
        psd_sample = _inverse_sampling_given_pdf(freqs, psd, sample_num = setting_dict['init_sample_num'])
        gmm = GaussianMixture(n_components=Num_Q, covariance_type='diag').fit(np.asarray(psd_sample).reshape(-1, 1))
        idx_thres = np.where(gmm.weights_ >= thres_hold)[0]
        
        

        if filename in ['CO2']:
            #SMKernel_hyp['weight'] = gmm.weights_[idx_thres].reshape(-1,1)    
            SMKernel_hyp['weight'] = np.ones(len(idx_thres)).reshape(-1,1)
            SMKernel_hyp['mean'] = gmm.means_[idx_thres].reshape(-1, setting_dict['input_dim'])
            SMKernel_hyp['mean_prior'] = .5*np.random.rand(Num_Q, setting_dict['input_dim'])
            SMKernel_hyp['std'] = np.sqrt(gmm.covariances_[idx_thres].reshape(-1, setting_dict['input_dim']))
            SMKernel_hyp['std_prior'] = .1*np.random.rand(Num_Q, setting_dict['input_dim'])


        elif filename in ['airline']:
            #SMKernel_hyp['weight'] = gmm.weights_[idx_thres].reshape(-1,1)    
            SMKernel_hyp['weight'] = np.ones(len(idx_thres)).reshape(-1,1)
            SMKernel_hyp['mean'] = gmm.means_[idx_thres].reshape(-1, setting_dict['input_dim'])
            SMKernel_hyp['mean_prior'] = .5*np.random.rand(Num_Q, setting_dict['input_dim'])
            SMKernel_hyp['std'] = np.sqrt(gmm.covariances_[idx_thres].reshape(-1, setting_dict['input_dim']))
            SMKernel_hyp['std_prior'] = .1*np.random.rand(Num_Q, setting_dict['input_dim'])

            
        else:
            #SMKernel_hyp['weight'] = 1 + gmm.weights_[idx_thres].reshape(-1,1) #for multioutput
            SMKernel_hyp['weight'] = np.ones(len(idx_thres)).reshape(-1,1)
            SMKernel_hyp['mean'] = gmm.means_[idx_thres].reshape(-1, setting_dict['input_dim'])
            SMKernel_hyp['mean_prior'] = .5*np.random.rand(Num_Q, setting_dict['input_dim'])
            SMKernel_hyp['std'] = np.sqrt(gmm.covariances_[idx_thres].reshape(-1, setting_dict['input_dim']))
            SMKernel_hyp['std_prior'] =.1*np.random.rand(Num_Q, setting_dict['input_dim'])
                        
            

        SMKernel_hyp['noise_variance'] = [1.0]   
        SMKernel_hyp['length_scale'] = [.5]
        setting_dict['hypparam'] = SMKernel_hyp

        return setting_dict

    





def _initialize_SMkernelhyp_uci( x_train,y_train, setting_dict, random_seed, yesSM = True, filename = None, model_name = None):
    """
    we consider initilaization method for High dimensional inputs empirically    
    """

    print('intialization by manually')
    np.random.seed(random_seed)
    SMKernel_hyp = {}
    N,D = list(x_train.shape)   
    num_Q = setting_dict['num_Q']
    #print(N,D)


    SMKernel_hyp['weight'] = np.ones(num_Q).reshape(-1, 1)
    if model_name in ['vssgp','weight_reg', 'weight_reg_nat', 'equal_reg', 'equal_reg_nat']:

        SMKernel_hyp['std'] = 0.05 + 0.45*np.random.rand(num_Q ,D) # real
        SMKernel_hyp['std_prior'] = 0.05*np.random.rand(num_Q ,D) # real
        SMKernel_hyp['mean'] = .25*np.random.rand(num_Q ,D) #regression task
        SMKernel_hyp['mean_prior'] = .05*np.random.rand(num_Q ,D)            
        
    else: #['vfegp','vssgp']
        SMKernel_hyp['std'] = 0.05 + 0.45*np.random.rand(num_Q ,D) # real
        SMKernel_hyp['std_prior'] = 0.5*np.random.rand(num_Q ,D) # real
        SMKernel_hyp['mean'] = .25*np.random.rand(num_Q ,D) #regression task
        SMKernel_hyp['mean_prior'] = .05*np.random.rand(num_Q ,D)    

    SMKernel_hyp['noise_variance'] = 1.0  # real v2
    SMKernel_hyp['variance'] = 1.0  # real v2    
    
    SMKernel_hyp['length_scale'] = 0.5 + 0.5*np.random.rand(D) # real    
    setting_dict['hypparam'] = SMKernel_hyp

    return setting_dict




def _make_gpmodel_v2(model_name,setting_dict,device,x_train,y_train):
    temp_model = _make_gpmodel(model_name, setting_dict, device)
    ith_model_name = temp_model.name
    if ith_model_name in ['gpvfe', 'gpvferbf', 'vssgp']:
        temp_model._set_data(batch_x=x_train, batch_y=y_train)
        if ith_model_name == 'vssgp':
            temp_model._set_inducing_pt(setting_dict['Num_Q'] * setting_dict['num_sample_pt'])
            optimizable_param = [*temp_model.parameters()]
        else:
            temp_model._set_inducing_pt(2 * setting_dict['Num_Q'] * setting_dict['num_sample_pt'])
            optimizable_param = [*temp_model.parameters(), temp_model.likelihood.variance]
    else:
        temp_model._set_data(x_train, y_train)
        optimizable_param = [*temp_model.parameters(), temp_model.likelihood.variance]
        
    if ith_model_name[-3:] == 'nat':
        print('variation adam optimizer \n')
        temp_optimizer = Adam_variation(optimizable_param,
                                        lr=setting_dict['lr_hyp'],
                                        betas=(0.9, 0.99),
                                        eps=1e-08,
                                        weight_decay=0.0)
    else:
        print('adam optimizer \n')
        temp_optimizer = torch.optim.Adam(optimizable_param,
                                          lr=setting_dict['lr_hyp'],
                                          betas=(0.9, 0.99),
                                          eps=1e-08,
                                          weight_decay=0.0)
    
    return temp_model,optimizable_param,temp_optimizer







def _make_gpmodel(model_name,setting_dict,device):
    if model_name == 'gprbf':
        param_dict = setting_dict['hypparam']
        Kern = RBF(variance = param_dict['variance'],
                   length_scale = param_dict['length_scale'],
                   device = device)
        model = gpr(Kern,
                    likelihood = None,
                    device = device,
                    param_dict=setting_dict)
        model.name = model_name
        return model


    if model_name == 'gpsm':
        param_dict = setting_dict['hypparam']
        Kern = SM(param_dict['weight'],
                  param_dict['mean'],
                  param_dict['std'],
                  device=device)

        model = gpr(Kern,
                    likelihood=None,
                    device = device,
                    param_dict=setting_dict)
        model.name = model_name
        return model

    if model_name == 'vssgp':
        param_dict = setting_dict['hypparam']
        model = vssgp(train_X=None,
                      train_Y=None,
                      setting_dict=setting_dict,
                      device= device)
        model.name = model_name
        return model


    if model_name == 'gpvferbf':
        param_dict = setting_dict['hypparam']
        Kern = RBF(variance = param_dict['variance'],
                   length_scale = param_dict['length_scale'],
                   device = device)

        model = gp_vfe(Kern,
                    likelihood=None,
                    device = device,
                    param_dict=setting_dict)
        model.name = model_name

        return model
    
    
    if model_name == 'gpvfe':
        param_dict = setting_dict['hypparam']
        Kern = SM(param_dict['weight'],
                  param_dict['mean'],
                  param_dict['std'],
                  device=device)

        model = gp_vfe(Kern,
                    likelihood=None,
                    device = device,
                    param_dict=setting_dict)
        model.name = model_name
        return model



    if model_name == 'rff':
        num_sample_pt = setting_dict['num_sample_pt']
        num_batch = setting_dict['num_batch']
        setting_dict['sampling_option'] = 'uniform'
        setting_dict['yes_nat'] = False

        print(ssgpr_sm(num_batch=num_batch,
                         num_sample_pt=num_sample_pt,
                         param_dict=setting_dict,
                         device=device) )
        
        
        model = ssgpr_sm(num_batch=num_batch,
                         num_sample_pt=num_sample_pt,
                         param_dict=setting_dict,
                         device=device)

        model.name = model_name
        return model

    

    if model_name == 'rffrp':
        num_sample_pt = setting_dict['num_sample_pt']
        num_batch = setting_dict['num_batch']
        setting_dict['sampling_option'] = 'uniform'
        setting_dict['yes_nat'] = False

        model = ssgpr_rep_sm_reg(num_batch=num_batch,
                                 num_sample_pt=num_sample_pt,
                                 param_dict=setting_dict,
                                 device=device)

        model.name = model_name
        return model

    

    if model_name == 'equal_reg':
        num_sample_pt = setting_dict['num_sample_pt']
        num_batch = setting_dict['num_batch']
        setting_dict['sampling_option'] = 'uniform'
        setting_dict['yes_nat'] = False

        model = ssgpr_rep_sm_reg(num_batch=num_batch,
                                 num_sample_pt=num_sample_pt,
                                 param_dict=setting_dict,
                                 device=device)

        model.name = model_name
        return model

    if model_name == 'naiveweight_reg':
        num_sample_pt = setting_dict['num_sample_pt']
        num_batch = setting_dict['num_batch']
        setting_dict['sampling_option'] = 'naive_weight'
        setting_dict['yes_nat'] = False

        model = ssgpr_rep_sm_reg(num_batch=num_batch,
                                 num_sample_pt=num_sample_pt,
                                 param_dict=setting_dict,
                                 device=device)
        model.name = model_name
        return model

    if model_name == 'weight_reg':
        num_sample_pt = setting_dict['num_sample_pt']
        num_batch = setting_dict['num_batch']
        setting_dict['sampling_option'] = 'weight'
        setting_dict['yes_nat'] = False

        model = ssgpr_rep_sm_reg(num_batch=num_batch,
                                 num_sample_pt=num_sample_pt,
                                 param_dict=setting_dict,
                                 device=device)
        model.name = model_name
        return model



    if model_name == 'equal_reg_nat':
        num_sample_pt = setting_dict['num_sample_pt']
        num_batch = setting_dict['num_batch']
        setting_dict['sampling_option'] = 'uniform'
        setting_dict['yes_nat'] = True

        model = ssgpr_rep_sm_reg(num_batch=num_batch,
                                 num_sample_pt=num_sample_pt,
                                 param_dict=setting_dict,
                                 device=device)
        model.name = model_name
        return model




    if model_name == 'weight_reg_nat':
        num_sample_pt = setting_dict['num_sample_pt']
        num_batch = setting_dict['num_batch']
        setting_dict['sampling_option'] = 'weight'
        setting_dict['yes_nat'] = True

        model = ssgpr_rep_sm_reg(num_batch=num_batch,
                                 num_sample_pt=num_sample_pt,
                                 param_dict=setting_dict,
                                 device=device)

        model.name = model_name
        return model

    return




if __name__ == "__main__":
    print(1)


