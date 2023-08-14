"""
A script for model training

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, Matthew O'Toole, G. Wetzstein.
SIGGRAPH 2022
"""
import os
import configargparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import utils
import params
import props.prop_model as prop_model
import image_loader as loaders
import torch
import os


# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add('--capture_subset', type=str, default=None)

params.add_parameters(p, 'train')
opt = params.set_configs(p.parse_args())
run_id = params.run_id_training(opt)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

if opt.gpu_id > 0:
#    torch.cuda.set_device(opt.gpu_id)
    print(f"Using gpu {opt.gpu_id} ...")

def main():
    if ',' in opt.data_path:
        opt.data_path = opt.data_path.split(',')
    else:
        opt.data_path = [opt.data_path]
    print(f'  - training a model ... Dataset path:{opt.data_path}')
    # Setup up dataloaders
    num_workers = 4
    # modify plane idxes!
    train_loader = DataLoader(loaders.PairsLoader([os.path.join(path, 'train') for path in opt.data_path],
                                                  plane_idxs=opt.plane_idxs['train'], image_res=opt.image_res,
                                                  avg_energy_ratio=opt.avg_energy_ratio, slm_type=opt.slm_type,
                                                  capture_subset=opt.capture_subset, dataset_subset=opt.dataset_subset),
                              num_workers=num_workers, batch_size=opt.batch_size, pin_memory=True)
    val_loader = DataLoader(loaders.PairsLoader([os.path.join(path, 'val') for path in opt.data_path],
                                                plane_idxs=opt.plane_idxs['train'], image_res=opt.image_res,
                                                shuffle=False, avg_energy_ratio=opt.avg_energy_ratio,
                                                slm_type=opt.slm_type, capture_subset=opt.capture_subset),
                            num_workers=num_workers, batch_size=opt.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(loaders.PairsLoader([os.path.join(path, 'test') for path in opt.data_path],
                                                 plane_idxs=opt.plane_idxs['all'], image_res=opt.image_res,
                                                 shuffle=False, avg_energy_ratio=opt.avg_energy_ratio, slm_type=opt.slm_type),
                             num_workers=num_workers, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    # Init model
    if opt.slm_type == 'ti':
        opt.roi_res = (760, 1240) # mofidy here!. should be 700, 1190?
    else:
        opt.roi_res = (840, 1200)
    model = prop_model.model(opt)
    model.train()

    # Init root path
    root_dir = os.path.join(opt.out_path, run_id)
    utils.cond_mkdir(root_dir)
    p.write_config_file(opt, [os.path.join(root_dir, 'config.txt')])

    psnr_checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="PSNR_validation_epoch", dirpath=root_dir,
                                                       filename="model-{epoch:02d}-{PSNR_validation_epoch:.2f}",
                                                       save_top_k=1, mode="max", )
    latest_checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="PSNR_validation_epoch", dirpath=root_dir,
                                                       filename="model-latest-{PSNR_validation_epoch:.2f}",
                                                       every_n_epochs=1, save_last=True)

    # Init trainer
    trainer = Trainer(default_root_dir=root_dir, accelerator='gpu',
                      log_every_n_steps=400, gpus=1, max_epochs=opt.num_epochs, callbacks=[psnr_checkpoint_callback, latest_checkpoint_callback])

    # Fit Model
    trainer.fit(model, train_loader, val_loader)
    
    # Test Model
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()