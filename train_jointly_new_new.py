from ovseg.data.JoinedData import JoinedData
import os
from ovseg.training.JoinedTraining import JoinedTraining
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
from ovseg.model.SegmentationModel import SegmentationModel
import pickle
import numpy as np
import torch
from tqdm import tqdm
from ovseg.utils import io


# %% collect basic infos for models
j = 0
val_fold = 0
data_name = 'OV04'
recon_model = ['recon_fbp_convs_full_HU', 'reconstruction_network_fbp_convs'][j]
proj_folder = ['projections_HU', 'projections'][j]
im_folder = ['images_HU_rescale', 'images_att'][j]
simulation = ['HU', 'win'][j]

for loss_weight in [0.5, 0.7, 0.9, 1.0]:

    # %% build data
    trn_dl_params = {'batch_size': 12, 'patch_size': [512, 512],
                     'num_workers': None, 'pin_memory': True,
                     'epoch_len': 250, 'store_coords_in_ram': True}
    val_dl_params = {'batch_size': 12, 'patch_size': [512, 512],
                     'num_workers': None, 'pin_memory': True,
                     'epoch_len': 25, 'store_coords_in_ram': True, 'store_data_in_ram': True,
                     'n_max_volumes': 50}
    preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed',
                                     data_name, 'pod_default')
    keys = ['projection', 'image', 'label', 'spacing']
    folders = [proj_folder, im_folder, 'labels', 'spacings']
    data = JoinedData(val_fold, preprocessed_path, keys, folders,
                      trn_dl_params=trn_dl_params,
                      val_dl_params=val_dl_params)
    # %% load models
    model1 = Reconstruction2dSimModel(val_fold, data_name, recon_model)
    model_path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                              data_name, 'pretrained_segmentation')
    model_params = pickle.load(open(os.path.join(model_path, 'model_parameters.pkl'), 'rb'))
    del model_params['augmentation']['GPU_params']['grayvalue']
    prep_params = pickle.load(open(os.path.join(preprocessed_path, 'preprocessing_parameters.pkl'),
                                   'rb'))
    model_params['preprocessing'] = prep_params
    model2 = SegmentationModel(val_fold, data_name, 'pretrained_segmentation',
                               model_parameters=model_params,
                               dont_store_data_in_ram=True)
    
    # %% opt and lr params
    opt1_params = {'lr': 0.5*10**-4, 'betas': (0.9, 0.999)}
    opt2_params = {'momentum': 0.99, 'weight_decay': 3e-5, 'nesterov': True,
                   'lr': 0.5**0.9 * 0.01}
    lr1_params = {'beta': 0.9, 'lr_min': 0}
    lr2_params = {'beta': 0.9, 'lr_min': 0}
    
    
    # %%
    model_path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                              data_name, 'joined_{:.1f}_{}'.format(loss_weight, simulation))
    training = JoinedTraining(model1, model2, data.trn_dl,  model_path,
                              loss_weight, num_epochs=500,
                              lr1_params=lr1_params, lr2_params=lr2_params,
                              opt1_params=opt1_params, opt2_params=opt2_params,
                              val_dl=data.val_dl)
    # %% now the magic!!
    training.train()
    
    # validation. A bit more complicated here. Other models need some improvements...
    results = {}
    val_path = os.path.join(model_path, 'validation')
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    val_scans = data.val_scans.copy()
    cases_save_img = val_scans[:10]
    print()
    print()
    for tpl, scan in tqdm(zip(data.val_ds, data.val_scans)):
        with torch.cuda.amp.autocast():
            recon = model1.predict(tpl, return_torch=True)
            recon_prep = model2.preprocessing(recon, tpl['spacing'])
            tpl['image'] = recon_prep
            pred = model2.predict(tpl, True)
        case_id = scan.split('.')[0]
        lb = tpl['label']
        if lb.max() > 0:
            results[case_id] = 200*np.sum(lb * pred) / np.sum(lb + pred)
        if scan in cases_save_img:
            recon_prep = recon_prep.cpu().numpy()
            io.save_nii(recon_prep, os.path.join(val_path, case_id+'_recon'),
                        model2.preprocessing.target_spacing)
            io.save_nii(pred, os.path.join(val_path, case_id+'_pred'),
                        model2.preprocessing.target_spacing)
    io.save_pkl(results, os.path.join(val_path, 'results_val'))
    
    dices = [results[key] for key in results]
    with open(os.path.join(val_path, 'results_val.txt'), 'wb') as outfile:
        outfile.write('Mean: {:.3f}, Median: {:.3f}'.format(np.nanmean(dices), np.nanmedian(dices)))
        for case in results:
            outfile.write(case + ': {:.3f}'.format(results[case]))
