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

from datasets.dataset import _load_collection_uci_data, _load_collection_bbq_data
from datasets.dataset_wilson_uci import _load_ucidataset_wilson

from utility.eval_metric import _evaluate_metric
from models_utility.construct_models import _initialize_SMkernelhyp_uci,_initialize_SMkernelhyp_uci_wilson, _make_gpmodel,_make_gpmodel_v2
from models_utility.personalized_adam import Adam_variation

device = True

print(torch.__version__)
print(torch.version.cuda)


save_format = '.pickle'
#pickle_savepath = './jupyters/result_pickle/'
pickle_savepath = './jupyters/result_pickle_ablation/'
# save_exp_path = './exp' + '/'
if not os.path.exists(pickle_savepath):



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




#datafilename_list = ['Concrete', 'skillcraft', 'parkinsons', 'kin8nm', 'elevators']
parser = argparse.ArgumentParser(description='data_file_load')
parser.add_argument('--filename', type=str, default='parkinsons')
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




    

comparison_model_name_list = ['rff','rffrp', 'equal_reg', 'equal_reg_nat', 'weight_reg', 'weight_reg_nat']


result_dict = {}
result_dict['Exp_setting'] = setting_dict
comparison_variable_numQ_list = [setting_dict['num_sample_pt']]
comparison_variable_sptnum_list = [setting_dict['num_sample_pt']]
comparison_variable_list = [args.lrhyp]  # lr
comparison_variable_kl_list = [setting_dict['kl_option']]



save_filename = expinfo + 'numcompmodel' + str(len(comparison_model_name_list))

for ith_Q in comparison_variable_numQ_list:
    setting_dict['Num_Q'] = ith_Q
    for ith_comparison_spt in comparison_variable_sptnum_list:
        setting_dict['num_sample_pt'] = ith_comparison_spt

        result_dict['Exp_setting'] = setting_dict

        result_dict['loss_list'] = {}
        result_dict['trmse_list'] = {}
        result_dict['tmnll_list'] = {}        
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
        for jth_rep in range(setting_dict['num_rep']):
            print('#' * 50)
            print('the following file running iteration %d th' % (jth_rep))
            print(save_filename0)
            print('')

            if args.filename in ['Concrete']:
                #x_train, x_test, y_train, y_test = _load_collection_bbq_data(args.filename,cuda_option=device)
                x_train, x_val,x_test, y_train, y_val, y_test = _load_collection_bbq_data(args.filename,cuda_option=device)
                
            else:
                if args.filename in ['parkinsons']:
                    x_train, x_val, x_test, y_train, y_val,y_test = _load_collection_uci_data(args.filename,
                                                                                             random_seed=args.randomseed + jth_rep,
                                                                                             numtotal=50000,
                                                                                             cuda_option=device,
                                                                                             normalize=args.datanormal)
                    
                    x_test = torch.cat([x_val,x_test],dim = 0)
                    y_test = torch.cat([y_val,y_test],dim = 0)

                else:
                    # parkinsons wrong and not reported in papers
                    x_train, x_test, y_train, y_test = _load_ucidataset_wilson(args.filename, split= .2 ,fold=jth_rep ,exp2 = True)
                    print('xtrain shape {}, xtest shape{}'.format(x_train.shape,x_test.shape))
                    print('ytrain shape {}, ytest shape{}'.format(y_train.shape,y_test.shape))

        
            
            setting_dict['input_dim'] = x_train.shape[1]
            setting_dict['noise_err'] = .05 * y_train.cpu().data.numpy().std()
           
            for ith_model_name in comparison_model_name_list:
                print('#' * 200)
                print(ith_model_name)
                
                best_loss = np.inf
                best_val = np.inf
                # initialization over 5 times
                num_init_rep = 5
                for ith_try in range(num_init_rep):
                    setting_dict = _initialize_SMkernelhyp_uci(x_train, y_train, setting_dict,args.randomseed + ith_try)  #manuall works better                    

                    temp_model, optimizable_param, temp_optimizer  = _make_gpmodel_v2(model_name=ith_model_name, setting_dict=setting_dict, device=device , x_train =x_train,y_train = y_train)
                  
                    try:
                        temp_model.train()
                        for i in range( num_init_rep):
                            temp_optimizer.zero_grad()
                            losstotal = temp_model.compute_loss(batch_x=x_train, batch_y=y_train,kl_option=setting_dict['kl_option'])
                            losstotal.backward()
                            temp_optimizer.step()
                            torch.cuda.empty_cache()
                        print('initialziation data shape {0}{1}'.format(x_train.shape,y_train.shape))                          
                        print('%d init loss: %.4f \n' % (ith_try, losstotal.cpu().data.numpy()))                                
                    except:
                        losstotal = torch.from_numpy(np.asarray(1e32)).cuda()
                        with open('./exp/main3_' + str(args.filename) + '_regression_task_Error_message.txt', 'a') as f:
                            f.write('initialization error model : {} in {} th experiment \n'.format(save_filename,i))
                            f.write('{}'.format(ith_model.name))                        
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
                train_rmse_list, train_mnll_list = [], []                
                rmse_list, mnll_list = [], []
                ermse_list, emnll_list = [], []
                
                
                for i in range(setting_dict['iter'] + 1):

                    ith_model.train()
                    optimizer.zero_grad()
                    ith_tic = time.time()
                    
                    
                    if ith_model_name in ['equal_reg', 'equal_reg_nat','weight_reg', 'weight_reg_nat']:
                        setting_dict['kl_option'] = True                    
                    else:
                        setting_dict['kl_option'] = False
                    
                    ith_loss = ith_model.compute_loss(batch_x=x_train, batch_y=y_train,kl_option=setting_dict['kl_option'])

                    ith_loss.backward()
                    optimizer.step()
                    ith_toc = time.time()

                    ###measurement_period = args.evalperiod
                    if i % args.evalperiod == 0:
                        ith_model.eval()            
                        if ith_model_name in ['rff','rffrp' ,'equal_reg', 'equal_reg_nat', 'weight_reg', 'weight_reg_nat']:
                            with torch.no_grad():
                                pred_train_mu, pred_train_var = ith_model._predict(inputs_new=x_train)                                                
                                pred_test_mu, pred_test_var = ith_model._predict(inputs_new=x_test)       
                                pred_eval_mu, pred_eval_var = pred_test_mu, pred_test_var #memory issue for elevators and other datasets

                            ith_rmse_train, ith_mnll_train = _evaluate_metric(pred_train_mu, pred_train_var, y_train)                                                                
                            ith_rmse, ith_mnll = _evaluate_metric(pred_test_mu, pred_test_var, y_test)                                
                            ith_ermse, ith_emnll = _evaluate_metric(pred_eval_mu, pred_eval_var, y_test)
                                
                        else:
                            with torch.no_grad():
                                pred_test_mu, pred_test_var = ith_model._predict(inputs_new=x_test)
                            ith_rmse, ith_mnll = _evaluate_metric(pred_test_mu, pred_test_var, y_test)                                                                
                            ith_ermse, ith_emnll = ith_rmse, ith_mnll
                                
                                
                        ith_time_list.append(ith_toc - ith_tic)
                        loss_list.append(ith_loss.cpu().data.numpy())
                        
                        train_rmse_list.append(ith_rmse_train)
                        rmse_list.append(ith_rmse)                        
                        train_mnll_list.append(ith_mnll_train)                        
                        mnll_list.append(ith_mnll)
                        
                        ermse_list.append(ith_ermse)
                        emnll_list.append(ith_emnll)                            
                        print('%d th loss: %.3f, tr mse : %.3f, val mse : %.3f, exact val mse : %.3f, tr mnll : %.3f, val mnll : %.3f, exact val mnll : %.3f' % 
                              (i, ith_loss.cpu().data.numpy(),ith_rmse_train, ith_rmse, ith_ermse,ith_mnll_train, ith_mnll,ith_emnll))                        


                    torch.cuda.empty_cache()
                    if best_val > ith_ermse:
                        best_val = ith_ermse
                        saved_param = save_model_param(temp_model)
                        print('parameters saved at {} iteration'.format(i))
                        

                torch.cuda.empty_cache()
                ith_model_saved = load_model_param(temp_model,saved_param)    
                ith_model_saved.eval()
                if ith_model_name in ['rff','rffrp' ,'equal_reg', 'equal_reg_nat', 'weight_reg', 'weight_reg_nat']:
                    with torch.no_grad():                    
                        pred_test_mu, pred_test_var = ith_model_saved._predict(inputs_new=x_test)                                                                
                        #pred_etest_mu, pred_etest_var = ith_model_saved._predict_exact(inputs_new=x_test)
                        pred_etest_mu, pred_etest_var = pred_test_mu, pred_test_var #memory issue for elevators and other datasets
                else: 
                    with torch.no_grad(): 
                        pred_test_mu, pred_test_var = ith_model_saved._predict(inputs_new=x_test)                                                                                    
                        pred_etest_mu, pred_etest_var = pred_test_mu, pred_test_var
                    
                    
                ith_rmse_t, ith_mnll_t = _evaluate_metric(pred_test_mu, pred_test_var, y_test)                                    
                ith_ermse_t, ith_emnll_t = _evaluate_metric(pred_etest_mu, pred_etest_var, y_test)

                print('\n')
                print('tmse : %.3f, exact tmse : %.3f, tmnll : %.3f, exact tmnll : %.3f' % ( ith_rmse_t, ith_ermse_t, ith_mnll_t,ith_emnll_t))                        
#                 print('vmse : %.3f, vmnll : %.3f' % ( ith_rmse_v, ith_mnll_v ))                        


                print('')
                if ith_model.name not in result_dict['loss_list'].keys():
                    # metric for validation
                    result_dict['loss_list'][ith_model.name] = [loss_list]
                    result_dict['trmse_list'][ith_model.name] = [train_rmse_list]
                    result_dict['tmnll_list'][ith_model.name] = [train_mnll_list]
                    
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
                    result_dict['trmse_list'][ith_model.name].append(train_rmse_list)
                    result_dict['tmnll_list'][ith_model.name].append(train_mnll_list)
                    
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

    
    
    result_dict['exp'] = args.__dict__
    with open(pickle_savepath + save_filename + '_ablation' + save_format, 'wb') as outfile:
        pickle.dump(result_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    print('-' * 50)
    print('saved file report')
    print('-' * 50)
    print('directory : {0}'.format(pickle_savepath))
    print('filename : {0}_ablation'.format(save_filename))
    print('\n' * 3)




