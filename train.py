import time
script_start_time = time.time()

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import warnings
from tqdm import tqdm

from data.triplet_sampler import *
from data.transform import ProportionalScalePad
from loss.losses import triplet_loss_fastreid
from lr_scheduler.sche_optim import make_optimizer, make_warmup_scheduler
from tensorboard_log import Logger
from processor import get_model, train_epoch, test_epoch 
from utils import count_parameters, set_modules_params_property, format_time, clear_cache, set_seed, get_train_data



warnings.filterwarnings(category=UserWarning, action="ignore")
warnings.filterwarnings(category=FutureWarning, action="ignore")
mp.set_sharing_strategy('file_system')


if __name__ == "__main__":
    data = get_train_data()

    alpha_ce= data['alpha_ce']
    beta_tri = data['beta_tri']

    # Set Seed for consistent and deterministic results
    set_seed(data['torch_seed'])

    tqdm.write(f"\nConfig used:\n\n{data}\n\nEnd config\n")

    # Transformation augmentation
    test_transform = transforms.Compose([
                      transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    #   ProportionalScalePad(target_width=data['x_length'], target_height=data['y_length']),
                      transforms.Normalize(data['n_mean'], data['n_std'])
    ])
                  
    train_base_transform = transforms.Compose(
        [
            transforms.Resize((data['y_length'],data['x_length']), antialias=True),
            # ProportionalScalePad(target_width=data['x_length'], target_height=data['y_length']),
            transforms.Pad(10)
        ]
    )
    train_random_transform = transforms.Compose([
                    transforms.RandomCrop((data['y_length'], data['x_length'])),
                    transforms.RandomHorizontalFlip(p=data['p_hflip']),
                    transforms.Normalize(data['n_mean'], data['n_std']),
                    transforms.RandomErasing(p=data['p_rerase'], value=0),
    ])        

    # Force a GPU
    if not data['parallel']:  
        if data['gpu']:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(data['gpu'])
    # Check if the GPU is available and select
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tqdm.write(f'Selected device: {device}')

    # Get data loaders
    if data['dataset'] == 'Veri776':
        start_time = time.time()
        tqdm.write('\nPreparing query dataset...')
        data_query = CUDADatasetVeri776Viewpoints(image_list=data['query_list_file'],
                                                  images_dir=data['images_dir'],
                                                  viewpoints=data['test_keypoints'],
                                                  is_train=False,
                                                  transform=test_transform,
                                                  device=data['preload_test_device'],
                                                  use_fp16=data['half_precision'],
                                                  preload_rate=data['preload_test_rate'],
                                                  preload_num_workers=data['preload_num_workers'],
                                                  preload_batch_size=data['preload_batch_size'],
                                                  )
        data_query = DataLoader(data_query,
                                batch_size=data['BATCH_SIZE'],
                                shuffle=False,
                                num_workers=data['num_workers_test'],
                                persistent_workers=True)

        tqdm.write('\nPreparing gallery dataset...')
        data_gallery = CUDADatasetVeri776Viewpoints(image_list=data['gallery_list_file'],
                                                    images_dir=data['images_dir'],
                                                    viewpoints=data['test_keypoints'],
                                                    is_train=False,
                                                    transform=test_transform,
                                                    device=data['preload_test_device'],
                                                    use_fp16=data['half_precision'],
                                                    preload_rate=data['preload_test_rate'],
                                                    preload_num_workers=data['preload_num_workers'],
                                                    preload_batch_size=data['preload_batch_size'],
                                                    )
        data_gallery = DataLoader(data_gallery,
                                  batch_size=data['BATCH_SIZE'],
                                  shuffle=False,
                                  num_workers=data['num_workers_test'],
                                  persistent_workers=True)

        tqdm.write('\nPreparing train dataset...')
        data_train = CUDADatasetVeri776(image_list=data['train_list_file'],
                                        images_dir=data['images_dir'],
                                        is_train=True,
                                        base_transform=train_base_transform,
                                        random_transform=train_random_transform,
                                        device=data['preload_train_device'],
                                        use_fp16=data['half_precision'],
                                        preload_rate=data['preload_train_rate'],
                                        preload_num_workers=data['preload_num_workers'],
                                        preload_batch_size=data['preload_batch_size'],
                                        )
        sampler = RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES'])
        data_train = DataLoader(data_train,
                                sampler=sampler,
                                num_workers=data['num_workers_train'],
                                batch_size=data['BATCH_SIZE'],
                                collate_fn=train_collate_fn,
                                persistent_workers=True)

        loading_time = time.time() - start_time
        tqdm.write(f'\nLoading datasets took {format_time(loading_time)}\n')

    clear_cache()

    # Create Model
    model = get_model(data, device)
    tqdm.write(f'\nModel \'{data["model_arch"]}\' with backbone \'{data["backbone"]}\' has {count_parameters(model) / 1e6:.1f}M trainable parameters')
    # Losses
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=data['label_smoothing'])
    metric_loss_fn = triplet_loss_fastreid(data['triplet_margin'], norm_feat=data['triplet_norm'], hard_mining=data['hard_mining'])

    # Optimizer
    optimizer = make_optimizer(data['optimizer'],
                               model,
                               data['lr'],
                               data['weight_decay'],
                               data['bias_lr_factor'],
                               data['momentum'])

    # Schedule for the optimizer           
    if data['epoch_freeze_L1toL3'] == 0:                 
        scheduler = make_warmup_scheduler(data['sched_name'],
                                          optimizer,
                                          data['num_epochs'],
                                          data['milestones'],
                                          data['gamma'],
                                          data['warmup_factor'],
                                          data['warmup_iters'],
                                          data['warmup_method'],
                                          last_epoch=-1,
                                          min_lr = data['min_lr']
                                          )
    else:
        scheduler = None

    # If running with fp16 precision a scaler is needed
    if data['half_precision']:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = False

    # Initiate a Logger with TensorBoard to store Scalars, Embeddings and the weights of the model
    logger = Logger(data)

    # Freeze backbone at warmupup epochs up to data['warmup_iters'] 
    if data['freeze_backbone_warmup']:
        set_modules_params_property((model.modelup2L3, model.modelL4), 'requires_grad', False)

    # Freeze up to the penultimate layer    
    if data['epoch_freeze_L1toL3'] > 0:
        set_modules_params_property((model.modelup2L3,), 'requires_grad', False)
        tqdm.write("\nFrozen backbone before branches!\n")
  
    # Training Loop
    for epoch in tqdm(range(data['num_epochs']),
                      desc='Training',
                      unit='epoch',
                      bar_format='{l_bar}{bar:80}{r_bar}',
                      position=0):
        # Unfreeze backbone
        if epoch == data['warmup_iters'] - 1: 
            set_modules_params_property((model.modelup2L3, model.modelL4), 'requires_grad', True)
     
        if epoch == data['epoch_freeze_L1toL3'] - 1:
            scheduler = make_warmup_scheduler(data['sched_name'],
                                              optimizer,
                                              data['num_epochs'],
                                              data,
                                              data['milestones'],
                                              data['gamma'],
                                              data['warmup_factor'],
                                              data['warmup_iters'],
                                              data['warmup_method'],
                                              last_epoch=-1,
                                              min_lr = data['min_lr'])

            set_modules_params_property((model.modelup2L3,), 'requires_grad', True)
            tqdm.write("\n\nUnfroze backbone before branches!")
        
        # Step schedule
        if epoch >= data['epoch_freeze_L1toL3'] - 1:
            scheduler.step()

        # Train Loop
        time_since_start = time.time() - script_start_time
        tqdm.write(f'\nTime since script start: {format_time(time_since_start)}')

        start_time = time.time()
        train_loss, ce_loss, triplet_loss, alpha_ce, beta_tri = train_epoch(model, device, data_train, ce_loss_fn, metric_loss_fn, optimizer, data, alpha_ce, beta_tri, logger, epoch, scheduler, scaler)
        tqdm.write(f'Total train epoch {epoch + 1} time: {format_time(time.time() - start_time)}')

        clear_cache()

        # Evaluation
        if epoch == 0 or (epoch + 1) % data['validation_period'] == 0 or epoch >= data['num_epochs'] - 15:
            tqdm.write('Start evaluation...')
            start_time = time.time()
            cmc, mAP = test_epoch(model, device, data_query, data_gallery, logger, epoch, remove_junk=True, scaler=scaler)
            tqdm.write(f'\nTotal evaluation time: {format_time(time.time() - start_time)}')
            tqdm.write(f'\nEpoch {epoch + 1}/{data["num_epochs"]}: Train Loss {train_loss:.4f} | CrossEntropy Loss {ce_loss:.4f} | Triplet Loss {triplet_loss:.4f} | Test mAP {mAP:.4f} | CMC1 {cmc[0]:.4f} | CMC5 {cmc[4]:.4f}\n')
            logger.save_model(model)

    tqdm.write(f"Best mAP: {max(logger.logscalars['Test/mAP'])}")
    tqdm.write(f"Best CMC1: {max(logger.logscalars['Test/CMC1'])}")
    tqdm.write(f"Best CMC5: {max(logger.logscalars['Test/CMC5'])}")
    logger.save_log()   
