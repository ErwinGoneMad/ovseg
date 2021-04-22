import os
import torch
import numpy as np


def get_model_params_2d_segmentation(n_classes=1,
                                     fp32=False):
    model_parameters = {}
    # we're doing no preprocessing parameters here as they can be loaded
    # from the preprocessed folder

    # Let's jump straight to the augmentation
    grayvalue_params = {'p_noise': 0.15,
                        'p_blur': 0.1,
                        'p_bright': 0.15,
                        'p_contr': 0.15,
                        'mm_var_noise': [0, 0.1],
                        'mm_sigma_blur': [0.5, 1.5],
                        'mm_bright': [0.7, 1.3],
                        'mm_contr': [0.65, 1.5],
                        'n_im_channels': 1
                        }
    spatial_params = {'p_rot': 0.2,
                      'p_zoom': 0.2,
                      'p_transl': 0,
                      'p_shear': 0,
                      'mm_zoom': [0.7, 1.4],
                      'mm_rot': [-180, 180],
                      'mm_transl': [-0.25, 0.25],
                      'mm_shear': [-0.2, 0.2],
                      'apply_flipping': True,
                      'n_im_channels': 1,
                      'out_shape': None}

    model_parameters['augmentation'] = {'torch_params': {'grid_inplane': spatial_params,
                                                         'grayvalue': grayvalue_params}}

    # now the network parameters. classic 2d UNet
    model_parameters['architecture'] = 'UNet'
    network_parameters = {'in_channels': 1,
                          'out_channels': n_classes + 1,
                          'kernel_sizes': 7 * [3],
                          'is_2d': True,
                          'filters': 32,
                          'filters_max': 320,
                          'n_pyramid_scales': None,
                          'conv_params': None,
                          'norm': None,
                          'norm_params': None,
                          'nonlin_params': None,
                          'kernel_sizes_up': None}
    model_parameters['network'] = network_parameters

    # we have no postprocessing parameters and just go with the default
    # now data
    ds_params = {}
    num_workers = 0 if os.name == 'nt' else 8
    trn_dl_params = {'patch_size': [512, 512],
                     'batch_size': 12,
                     'num_workers': num_workers,
                     'pin_memory': torch.cuda.is_available(),
                     'epoch_len': 250,
                     'p_bias_sampling': 0,
                     'min_biased_samples': 3,
                     'padded_patch_size': None,
                     'store_coords_in_ram': True,
                     'memmap': 'r',
                     'n_im_channels': 1,
                     'store_data_in_ram': False,
                     'return_fp16': not fp32,
                     'n_max_volumes': None}
    val_dl_params = trn_dl_params.copy()
    val_dl_params['epoch_len'] = 25
    val_dl_params['store_data_in_ram'] = True
    val_dl_params['n_max_volumes'] = 50
    keys = ['image', 'label']
    folders = ['images', 'labels']
    data_params = {'n_folds': 4,
                   'fixed_shuffle': True,
                   'ds_params': ds_params,
                   'trn_dl_params': trn_dl_params,
                   'keys': keys,
                   'val_dl_params': val_dl_params,
                   'folders': folders}
    model_parameters['data'] = data_params

    # prediction object
    prediction_params = {'patch_size': [512, 512],
                         'batch_size': 1,
                         'overlap': 0.5,
                         'fp32': fp32,
                         'patch_weight_type': 'linear',
                         'linear_min': 0.1,
                         'mode': 'flip'}
    model_parameters['prediction'] = prediction_params

    # now finally the training!
    loss_params = {'eps': 1e-5,
                   'dice_weight': 1.0,
                   'ce_weight': 1.0,
                   'pyramid_weight': 0.5}
    opt_params = {'momentum': 0.99, 'weight_decay': 3e-5, 'nesterov': True,
                  'lr': 10**-2}
    lr_params = {'beta': 0.9, 'lr_min': 0}
    training_params = {'loss_params': loss_params,
                       'num_epochs': 1000, 'opt_params': opt_params,
                       'lr_params': lr_params, 'nu_ema_trn': 0.99,
                       'nu_ema_val': 0.7, 'fp32': fp32,
                       'p_plot_list': [1, 0.5, 0.2], 'opt_name': 'SGD'}
    model_parameters['training'] = training_params
    model_parameters['prediction_key'] = 'learned_segmentation'
    return model_parameters


def get_model_params_3d_nnUNet(patch_size, n_2d_convs, n_classes=1, fp32=False):
    model_params = get_model_params_2d_segmentation(n_classes=n_classes,
                                                    fp32=fp32)

    # first determine the number of stages plus the kernel sizes used there
    i = 1
    kernel_sizes = [(1, 3, 3) if 0 < n_2d_convs else (3, 3, 3)]
    psb = np.array(patch_size).copy()
    while psb.max() >= 8:
        j = 1 if kernel_sizes[-1][0] == 1 else 0
        psb[j:] = psb[j:] // 2
        kernel_sizes.append((1, 3, 3) if len(kernel_sizes) < n_2d_convs else (3, 3, 3))
        i += 1

    kernel_sizes_up = []
    for i in range(len(kernel_sizes) - 1):
        if kernel_sizes[i] == (1, 3, 3) and kernel_sizes[i+1] == (1, 3, 3):
            kernel_sizes_up.append((1, 3, 3))
        else:
            kernel_sizes_up.append((3, 3, 3))

    model_params['network']['kernel_sizes'] = kernel_sizes
    model_params['network']['is_2d'] = False
    model_params['network']['use_3d_convs_when_upsampling'] = True

    padded_patch_size = patch_size.copy()
    padded_patch_size[1] = padded_patch_size[1] * 2
    padded_patch_size[2] = padded_patch_size[2] * 2

    model_params['augmentation']['torch_params']['grid_inplane']['out_shape'] = patch_size

    # next change the dataloader arguments
    for key in ['trn_dl_params', 'val_dl_params']:
        model_params['data'][key]['patch_size'] = patch_size
        model_params['data'][key]['batch_size'] = 2
        model_params['data'][key]['min_biased_samples'] = 1
        model_params['data'][key]['padded_patch_size'] = padded_patch_size

    model_params['prediction']['patch_size'] = patch_size

    return model_params
