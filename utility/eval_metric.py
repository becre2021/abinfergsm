import torch
import numpy as np

def _evaluate_metric(pred_mu, pred_var, y_test):
    #out1 : root mean square (rmse)
    #out2 : mean negative log likelihood (mnll)
    with torch.no_grad():
        out1 = ((pred_mu - y_test).pow(2).mean()).sqrt()
        
        log_part = ((2 * np.pi * pred_var).log()).div(2.0)
        unc_part = ((pred_mu - y_test).pow(2)).div(2*pred_var + 1e-16)
        summed_parts = log_part + unc_part
        out2 = summed_parts.mean()

    return out1.cpu().data.numpy(), out2.cpu().data.numpy()


# def _write_summary(save_filename,result_dict):
#     with open('./jupyters/experiment_result_summary.txt', 'a') as result_summary:
#         result_summary.write('#' * 100 + '\n')
#         result_summary.write('experiment setting \n')
#         result_summary.write('%s \n' % (save_filename))
#         result_summary.write('\n')

#         result_summary.write('loss' + '\n')
#         for ith_model_name in result_dict['best_loss']:
#             result_summary.write(ith_model_name + '\n')
#             result_summary.write('{} \n'.format(np.asarray(result_dict['best_loss'][ith_model_name])))
#             result_summary.write('mean : {0:.4f} \n'.format(np.asarray(result_dict['best_loss'][ith_model_name]).mean()))
#             result_summary.write('std : {0:.4f} \n'.format(np.asarray(result_dict['best_loss'][ith_model_name]).std()))
#         result_summary.write('\n')

#         result_summary.write('rmse' + '\n')
#         for ith_model_name in result_dict['best_rmse']:
#             result_summary.write(ith_model_name + '\n')
#             result_summary.write('{} \n'.format(np.asarray(result_dict['best_rmse'][ith_model_name])))
#             result_summary.write('mean : {0:.4f} \n'.format(np.asarray(result_dict['best_rmse'][ith_model_name]).mean()))
#             result_summary.write('std : {0:.4f} \n'.format(np.asarray(result_dict['best_rmse'][ith_model_name]).std()))
#         result_summary.write('\n')

#         result_summary.write('mnll' + '\n')
#         for ith_model_name in result_dict['best_mnll']:
#             result_summary.write(ith_model_name + '\n')
#             result_summary.write('{} \n'.format(np.asarray(result_dict['best_mnll'][ith_model_name])))
#             result_summary.write('mean : {0:.4f} \n'.format(np.asarray(result_dict['best_mnll'][ith_model_name]).mean()))
#             result_summary.write('std : {0:.4f} \n'.format(np.asarray(result_dict['best_mnll'][ith_model_name]).std()))
#         result_summary.write('\n')

#         result_summary.write('\n')
#         result_summary.write('\n')

#     return