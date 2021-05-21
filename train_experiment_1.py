from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)

args = parser.parse_args()
p_name = 'pod_half'

val_fold_list = list(range(5, 8))
exp_list = 3 * [args.gpu]


def get_model_params(exp):
    assert exp in [0, 1, 2], "experiment must be 0, 1 or 2"
    prg_trn_sizes = [[16, 64*2, 64*2],
                     [20, 80*2, 80*2],
                     [24, 96*2, 96*2],
                     [28, 112*2, 112*2],
                     [32, 128*2, 128*2]]
    out_shape = [[16, 64, 64],
                 [20, 80, 80],
                 [24, 96, 96],
                 [28, 112, 112],
                 [32, 128, 128]]
    if exp == 1:
        out_shape = out_shape[:-1]
        prg_trn_sizes = prg_trn_sizes[:-1]
    elif exp == 2:
        out_shape = out_shape[:-2]
        prg_trn_sizes = prg_trn_sizes[:-2]

    patch_size = out_shape[-1]
    model_name = '5_stages_UNet_{}_{}'.format(patch_size[0], patch_size[1])

    model_params = get_model_params_3d_nnUNet(patch_size, 2, use_prg_trn=True)
    # first change the UNet in the way we want to have it
    model_params['network']['n_pyramid_scales'] = 4
    model_params['network']['kernel_sizes'] = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3),
                                               (3, 3, 3)]
    model_params['network']['kernel_sizes_up'] = [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    model_params['training']['prg_trn_sizes'] = prg_trn_sizes
    # this time we change the amount of augmentation during training
    prg_trn_aug_params = {}
    # params = model_params['augmentation']['torch_params']['grayvalue']
    # for key in params:
    #     if key.startswith('p'):
    #         prg_trn_aug_params[key] = [params[key]/2, params[key]]
    # factor we use for reducing the magnitude of the gray value augmentations
    c = 5
    prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
    prg_trn_aug_params['mm_sigma_blur'] = np.array([[1 - 0.5/c, 1 + 0.5/c], [0.5, 1.5]])
    prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
    prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.45/c, 1 + 0.5/c], [0.65, 1.5]])
    prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
    prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
    prg_trn_aug_params['out_shape'] = out_shape
    # params = model_params['augmentation']['torch_params']['grid_inplane']
    # for key in params:
    #     if key.startswith('p'):
    #         prg_trn_aug_params[key] = [params[key]/2, params[key]]
    model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
    model_params['training']['prg_trn_resize_on_the_fly'] = False
    return model_params, model_name


for val_fold, exp in zip(val_fold_list, exp_list):
    model_params, model_name = get_model_params(exp)
    model = SegmentationModel(val_fold=val_fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()
    del model.network
    for tpl in model.data.val_dl.dataset.data:
        for arr in tpl:
            del arr
        del tpl

ens = SegmentationEnsemble(val_fold=list(range(5, 8)),
                           data_name='OV04',
                           preprocessed_name=p_name,
                           model_name=model_name)
if ens.all_folds_complete():
    ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
