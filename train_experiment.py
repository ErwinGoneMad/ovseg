from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

model_name = 'cubed_patches'
if int(args.i) == 0:
    p_name = 'pod_half'
    val_fold = 3
elif int(args.i) == 1:
    p_name = 'pod_half'
    val_fold = 4
elif int(args.i) == 2:
    p_name = 'om_half'
    val_fold = 3
elif int(args.i) == 3:
    p_name = 'om_half'
    val_fold = 4


# model_params = get_model_params_3d_nnUNet([48, 192, 192], 2,
#                                           use_prg_trn=use_prg_trn)

model = SegmentationModel(val_fold=val_fold,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name=model_name)
model.training.train()
model.eval_validation_set()
del model.network
for tpl in model.data.val_dl.dataset.data:
    for arr in tpl:
        del arr
    del tpl

ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=p_name,
                           model_name=model_name)

if ens.all_folds_complete():
    ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
