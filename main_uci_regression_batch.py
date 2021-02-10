# train : 80% , validation :10%,  test : 10%
import copy
import os
import argparse
import copy
import torch
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime

from datasets.dataset import _load_collection_uci_data, _load_collection_uci_data_batch,  _traintestsplit
from datasets.dataset_wilson_uci import _load_ucidataset_wilson,_load_ucidataset_wilson_batch
from models_utility.construct_models import _initialize_SMkernelhyp_uci,_initialize_SMkernelhyp_uci_wilson,_make_gpmodel_v2
from models_utility.personalized_adam import Adam_variation
from utility.eval_metric import _evaluate_metric



device = True
print(torch.__version__)
print(torch.version.cuda)


save_format = '.pickle'
pickle_savepath = './jupyters/result_UCIregressionBatch/'
if not os.path.exists(pickle_savepath):
    os.makedirs(pickle_savepath)


    
def make_expinfo(args):
    args_dict = args.__dict__
    model_info = ''
    for ith_keys in args_dict:
        model_info += ith_keys + str(args_dict[ith_keys]) + '_'
    return model_info


def save_model_param(temp_model):
    param_dict = {}
    for ith_name,ith_param in temp_model.named_parameters():
        param_dict[ith_name] = ith_param.data.clone()    
    if hasattr(temp_model,'likelihood'):
        param_dict['var_noise'] = temp_model.likelihood.variance.data.clone()     
    return param_dict


def load_model_param(temp_model,saved_param):
    for ith_name,ith_param in temp_model.named_parameters():
        ith_param.data = saved_param[ith_name]        
    if hasattr(temp_model,'likelihood'):
        temp_model.likelihood.variance.data = saved_param['var_noise']
    return temp_model




# CUDA_VISIBE_DEVICES=3 python3 .py --filname --numQ --numspt --numbatch 1 --ratesamplespt .05 --lrhyp .005 --numrepexp 5


parser = argparse.ArgumentParser(description='data_file_load')
parser.add_argument('--filename', type=str, default='elevators')
parser.add_argument('--numQ', type=int, default=4)
parser.add_argument('--numspt', type=int, default=10)
parser.add_argument('--numtotalspt', type=int, default=1)
parser.add_argument('--numbatch', type=int, default=3)
parser.add_argument('--ratesamplespt', type=float, default=.05)
parser.add_argument('--lrhyp', type=float, default=.005)
parser.add_argument('--iter', type=int, default=500)
parser.add_argument('--numrepexp', type=int, default=2)
parser.add_argument('--numinitsample', type=int, default=500)
parser.add_argument('--randomseed', type=int, default=1111)
parser.add_argument('--kloption', type=bool, default=True)
parser.add_argument('--datanormal', type=bool, default=True)
parser.add_argument('--evalperiod', type=int, default=50)



args = parser.parse_args()
args.numtotalspt = args.numQ * args.numspt
expinfo = make_expinfo(args)

setting_dict = {}
setting_dict['filename'] = args.filename
setting_dict['Num_Q'] = args.numQ
setting_dict['num_sample_pt'] = args.numspt
setting_dict['num_total_pt'] = args.numtotalspt
setting_dict['num_batch'] = args.numbatch
setting_dict['lr_hyp'] = args.lrhyp
setting_dict['iter'] = args.iter
setting_dict['num_rep'] = args.numrepexp
setting_dict['init_sample_num'] = args.numinitsample
setting_dict['random_seed'] = args.randomseed
setting_dict['weight_rate'] = args.ratesamplespt
setting_dict['kl_option'] = args.kloption

print('expinfo')
print(expinfo)




    

comparison_model_name_list = ['gpvferbf','gpvfe','rff','vssgp', 'equal_reg', 'equal_reg_nat', 'weight_reg', 'weight_reg_nat']



result_dict = {}
result_dict['Exp_setting'] = setting_dict
comparison_variable_numQ_list = [setting_dict['num_sample_pt']] 
comparison_variable_sptnum_list = [setting_dict['num_sample_pt']]
comparison_variable_list = [args.lrhyp]  # lr
comparison_variable_kl_list = [setting_dict['kl_option']] # kl



# num_Q x num_Spt
#x_full_list, y_full_list= _load_collection_uci_data_v2(args.filename, random_seed=args.randomseed, numtotal=60000, cuda_option=device, normalize=args.datanormal,numrep = args.numrepexp)
x_full_train, x_full_val, x_full_test, y_full_train, y_full_val,y_full_test = _load_ucidataset_wilson_batch(args.filename, split= .2 ,fold=0)
print('x_full_train.shape, x_full_val.shape , x_full_test.shape') 
print(x_full_train.shape, x_full_val.shape , x_full_test.shape) 

save_filename = expinfo + 'numcompmodel' + str(len(comparison_model_name_list))
for ith_Q in comparison_variable_numQ_list:
    setting_dict['Num_Q'] = ith_Q
    for ith_comparison_spt in comparison_variable_sptnum_list:
        setting_dict['num_sample_pt'] = ith_comparison_spt

        result_dict['Exp_setting'] = setting_dict

        result_dict['loss_list'] = {}
        result_dict['rmse_list'] = {}
        result_dict['mnll_list'] = {}
        result_dict['ermse_list'] = {}
        result_dict['emnll_list'] = {}

        result_dict['rmse'] = {}
        result_dict['mnll'] = {}
        result_dict['ermse'] = {}
        result_dict['emnll'] = {}

        result_dict['pred_test_mu'] = {}
        result_dict['pred_test_var'] = {}

        result_dict['param_history'] = {}
        result_dict['error_history'] = {}
        result_dict['time_list'] = {}
        
        save_filename0 = expinfo
        for jth_rep in range(args.numrepexp):
            print('#' * 50)
            print('the following file running iteration %d th' % (jth_rep))
            print(save_filename0)
            print('')


            x_train, x_val, x_test = x_full_train[jth_rep::args.numrepexp], x_full_val[jth_rep::args.numrepexp], x_full_test[jth_rep::args.numrepexp]
            y_train, y_val, y_test = y_full_train[jth_rep::args.numrepexp], y_full_val[jth_rep::args.numrepexp], y_full_test[jth_rep::args.numrepexp]
            
            setting_dict['input_dim'] = x_train.shape[1]
            setting_dict['noise_err'] =  0.05*y_train.cpu().data.numpy().std()           
            
            
            for ith_model_name in comparison_model_name_list:
                print('#' * 200)
                print(ith_model_name)
                print('xtrain shape {}, xval shape{}, xtest shape{}'.format(x_train.shape,x_val.shape,x_test.shape))
                print('ytrain shape {}, yval shape{}, ytest shape{}'.format(y_train.shape,y_val.shape,y_test.shape))
                
                best_loss = np.inf
                best_val = np.inf
                # initialization over 5 times due to memory issue
                num_init_rep = 5
                for ith_try in range(num_init_rep):
                    setting_dict = _initialize_SMkernelhyp_uci(x_train, y_train, setting_dict,args.randomseed + ith_try)          
                    
                    temp_model, optimizable_param, temp_optimizer  = _make_gpmodel_v2(model_name=ith_model_name, setting_dict=setting_dict, device=device , x_train =x_train,y_train = y_train)                
                    try:
                        temp_model.train()
                        for i in range(100 + 1):
                            temp_optimizer.zero_grad()
                            losstotal = temp_model.compute_loss(batch_x=x_train[ith_try::num_init_rep], batch_y=y_train[ith_try::num_init_rep],kl_option=setting_dict['kl_option'])
                            
                            losstotal.backward()
                            temp_optimizer.step()
                            torch.cuda.empty_cache()
                        print('initialziation data shape {0}{1}'.format(x_train.shape,y_train.shape))                                                  
                        print('%d init loss: %.4f \n' % (ith_try, losstotal.cpu().data.numpy()))                                
                    except:
                        losstotal = torch.from_numpy(np.asarray(1e32)).cuda()
                        with open('./exp/main3_' + str(args.filename) + '_regression_task_Error_message.txt', 'a') as f:
                            f.write('initialization error model : {} in {} th experiment \n'.format(save_filename,i))
                            f.write('{}'.format(temp_model.name))                        
                            f.write('\n\n')                                                    
                        pass
                    
                if best_loss >= losstotal.cpu().data.numpy():
                    best_loss = losstotal.cpu().data.numpy()
                    saved_param = save_model_param(temp_model)
                    ith_model = load_model_param(temp_model,saved_param)
                    optimizer = temp_optimizer
                                        
                    print('%d model chosen ' % (ith_try))
                else:
                    pass
                torch.cuda.empty_cache()

                ####################################################################################################################################
                ####################################################################################################################################
                #try:
                ##### metric setup
                loss_list, ith_time_list = [], []
                rmse_list, mnll_list = [], []
                ermse_list, emnll_list = [], []
                
                
                if ith_model_name in ['gpvferbf','gpvfe']:
                    #setting_dict['iter'] = 1500 # 1000 이상 overfitting check
                    setting_dict['iter'] = args.iter                                    
                else:
                    setting_dict['iter'] = args.iter                
                for i in range(setting_dict['iter'] + 1):

                    ith_model.train()
                    optimizer.zero_grad()
                    ith_tic = time.time()
                    ith_loss = ith_model.compute_loss(batch_x=x_train, batch_y=y_train,kl_option=setting_dict['kl_option'])

                    ith_loss.backward()
                    optimizer.step()
                    ith_toc = time.time()

                    ###measurement_period = args.evalperiod
                    if i % args.evalperiod == 0:
                        ith_model.eval()            
                        if ith_model_name in ['rff', 'weight_reg', 'weight_reg_nat', 'equal_reg', 'equal_reg_nat']:
                            with torch.no_grad():
                                pred_val_mu, pred_val_var = ith_model._predict(inputs_new=x_val)                                                
                                pred_eval_mu, pred_eval_var = ith_model._predict_exact(inputs_new=x_val)
                            ith_rmse, ith_mnll = _evaluate_metric(pred_val_mu, pred_val_var, y_val)                                
                            ith_ermse, ith_emnll = _evaluate_metric(pred_eval_mu, pred_eval_var, y_val)
                                
                        else:
                            with torch.no_grad():
                                pred_val_mu, pred_val_var = ith_model._predict(inputs_new=x_val)
                            ith_rmse, ith_mnll = _evaluate_metric(pred_val_mu, pred_val_var, y_val)                                                                
                            ith_ermse, ith_emnll = ith_rmse, ith_mnll
                                
                                
                        ith_time_list.append(ith_toc - ith_tic)
                        loss_list.append(ith_loss.cpu().data.numpy())
                        rmse_list.append(ith_rmse)
                        mnll_list.append(ith_mnll)
                        ermse_list.append(ith_ermse)
                        emnll_list.append(ith_emnll)                            
                        print('%d th loss: %.3f, val mse : %.3f, exact val mse : %.3f, val mnll : %.3f, exact val mnll : %.3f' % (i, ith_loss.cpu().data.numpy(), ith_rmse, ith_ermse, ith_mnll,ith_emnll))                        
                        

                    torch.cuda.empty_cache()
                    if best_val > ith_ermse:
                        best_val = ith_ermse
                        saved_param = save_model_param(temp_model)
                        print('parameters saved at {} iteration'.format(i))
                        

                torch.cuda.empty_cache()
                #### test evalutation
                ith_model_saved = load_model_param(temp_model,saved_param)    
                ith_model_saved.eval()
                if ith_model_name in ['rff', 'weight_reg', 'weight_reg_nat', 'equal_reg', 'equal_reg_nat']:
                    with torch.no_grad():                    
                        pred_test_mu, pred_test_var = ith_model_saved._predict(inputs_new=x_test)                                                                
                        pred_etest_mu, pred_etest_var = ith_model_saved._predict_exact(inputs_new=x_test)
                    
                else:
                    with torch.no_grad(): 
                        pred_test_mu, pred_test_var = ith_model_saved._predict(inputs_new=x_test)                                                                                    
                        pred_etest_mu, pred_etest_var = pred_test_mu, pred_test_var
                    
                    
                ith_rmse_t, ith_mnll_t = _evaluate_metric(pred_test_mu, pred_test_var, y_test)                                    
                ith_ermse_t, ith_emnll_t = _evaluate_metric(pred_etest_mu, pred_etest_var, y_test)

                print('\n')
                print('tmse : %.3f, exact tmse : %.3f, tmnll : %.3f, exact tmnll : %.3f' % ( ith_rmse_t, ith_ermse_t, ith_mnll_t,ith_emnll_t))                        


                print('')
                if ith_model.name not in result_dict['loss_list'].keys():
                    # metric for validation
                    result_dict['loss_list'][ith_model.name] = [loss_list]
                    result_dict['rmse_list'][ith_model.name] = [rmse_list]
                    result_dict['mnll_list'][ith_model.name] = [mnll_list]
                    result_dict['ermse_list'][ith_model.name] = [ermse_list]
                    result_dict['emnll_list'][ith_model.name] = [emnll_list]
                    # metric for testset
                    result_dict['rmse'][ith_model.name] = [ith_rmse_t]
                    result_dict['mnll'][ith_model.name] = [ith_mnll_t]
                    result_dict['ermse'][ith_model.name] = [ith_ermse_t]
                    result_dict['emnll'][ith_model.name] = [ith_emnll_t]
                    result_dict['time_list'][ith_model.name] = [np.asarray(ith_time_list).mean()]

                else:
                    # metric for validation                    
                    result_dict['loss_list'][ith_model.name].append(loss_list)
                    result_dict['rmse_list'][ith_model.name].append(rmse_list)
                    result_dict['mnll_list'][ith_model.name].append(mnll_list)
                    result_dict['ermse_list'][ith_model.name].append(ermse_list)
                    result_dict['emnll_list'][ith_model.name].append(emnll_list)
                    # metric for testset
                    result_dict['rmse'][ith_model.name].append(ith_rmse_t)
                    result_dict['mnll'][ith_model.name].append(ith_mnll_t)
                    result_dict['ermse'][ith_model.name].append(ith_ermse_t)
                    result_dict['emnll'][ith_model.name].append(ith_emnll_t)
                    result_dict['time_list'][ith_model.name].append(np.asarray(ith_time_list).mean())
                        

    print('result_dict[ermse]')
    print(result_dict['ermse'])
    result_static = {}
    result_static['exp'] = args.__dict__
    result_static['result_dict'] = result_dict
    
    for ith, ith_model in enumerate(result_dict['ermse']):
        mean0 = np.asarray(result_dict['rmse'][ith_model]).mean().round(4)
        std0 = np.asarray(result_dict['rmse'][ith_model]).std().round(4)
        mean1 = np.asarray(result_dict['ermse'][ith_model]).mean().round(4)
        std1 = np.asarray(result_dict['ermse'][ith_model]).std().round(4)

        mean2 = np.asarray(result_dict['mnll'][ith_model]).mean().round(4)
        std2 = np.asarray(result_dict['mnll'][ith_model]).std().round(4)
        mean3 = np.asarray(result_dict['emnll'][ith_model]).mean().round(4)
        std3 = np.asarray(result_dict['emnll'][ith_model]).std().round(4)

        mean4 = np.asarray(result_dict['time_list'][ith_model]).mean().round(4)
        std4 = np.asarray(result_dict['time_list'][ith_model]).std().round(4)

        if ith == 0:
            result_static['rmse_mean'], result_static['rmse_std'] = {}, {}
            result_static['ermse_mean'], result_static['ermse_std'] = {}, {}
            result_static['mnll_mean'], result_static['mnll_std'] = {}, {}
            result_static['emnll_mean'], result_static['emnll_std'] = {}, {}
            result_static['time_mean'], result_static['time_std'] = {}, {}

            result_static['rmse_mean'][ith_model] = mean0
            result_static['rmse_std'][ith_model] = std0
            result_static['ermse_mean'][ith_model] = mean1
            result_static['ermse_std'][ith_model] = std1

            result_static['mnll_mean'][ith_model] = mean2
            result_static['mnll_std'][ith_model] = std2
            result_static['emnll_mean'][ith_model] = mean3
            result_static['emnll_std'][ith_model] = std3

            result_static['time_mean'][ith_model] = mean4
            result_static['time_std'][ith_model] = std4

        else:
            result_static['rmse_mean'][ith_model] = mean0
            result_static['rmse_std'][ith_model] = std0
            result_static['ermse_mean'][ith_model] = mean1
            result_static['ermse_std'][ith_model] = std1

            result_static['mnll_mean'][ith_model] = mean2
            result_static['mnll_std'][ith_model] = std2
            result_static['emnll_mean'][ith_model] = mean3
            result_static['emnll_std'][ith_model] = std3

            result_static['time_mean'][ith_model] = mean4
            result_static['time_std'][ith_model] = std4

    with open(pickle_savepath + save_filename + '_static' + save_format, 'wb') as outfile:
        pickle.dump(result_static, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    print('-' * 50)
    print('saved file report')
    print('-' * 50)
    print('directory : {0}'.format(pickle_savepath))
    print('filename : {0}_static'.format(save_filename))
    print('\n' * 3)

    save_filename = expinfo + 'numcompmodel' + str(len(comparison_model_name_list))
    with open(pickle_savepath + save_filename + save_format, 'wb') as outfile:
        pickle.dump(result_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    print('-' * 50)
    print('saved file report')
    print('-' * 50)
    print('directory : {0}'.format(pickle_savepath))
    print('filename : {0}'.format(save_filename))
    print('\n' * 3)

    with open('./exp/main3_' + str(args.filename) + '_regression_task_revision_v2.txt', 'a') as f:
        f.write('#' * 100 + '\n')
        f.write('%s \n' % (save_filename))
        f.write('\n')
        for ith_key in ['rmse', 'ermse', 'mnll', 'emnll', 'time']:
            f.write('%s %s \n' % (ith_key + '_mean', ith_key + '_std'))
            f.write('%s \n' % (result_static[ith_key + '_mean']))
            f.write('%s \n' % (result_static[ith_key + '_std']))
            f.write('\n')
        f.write('\n' * 3)

        
        
        
        
        
        
        
       
