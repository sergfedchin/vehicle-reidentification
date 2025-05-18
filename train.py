import time
script_start_time = time.time()

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import argparse
import yaml
import os
import warnings
import gc
from tqdm import tqdm

from data.triplet_sampler import *
from loss.losses import triplet_loss_fastreid
from lr_scheduler.sche_optim import make_optimizer, make_warmup_scheduler
from tensorboard_log import Logger
from processor import get_model, train_epoch, test_epoch 


mp.set_sharing_strategy('file_system')


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(t: int) -> str:
    return f'{f"{int(t // 3600)}h " if t >= 3600 else ""}'\
           f'{f"{int(t // 60) % 60}m " if t >= 60 else ""}'\
           f'{f"{int(t % 60)}" if t >= 10 else f"{t:.1f}"}s'


def clear_cache():
    print('Emptying cache...', end=' ')
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    cache_emptying_time = time.time() - start_time
    print(f'Took {format_time(cache_emptying_time)}\n')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ReID model trainer')
    parser.add_argument('--config', default=None, help='Config Path')
    parser.add_argument('--batch_size', default=None, type=int, help='Batch size')
    parser.add_argument('--backbone', default=None, help='Model Backbone')
    parser.add_argument('--hflip', default=None, type=float, help='Probabilty for horizontal flip')
    parser.add_argument('--randomerase', default=None, type=float,  help='Probabilty for random erasing')
    parser.add_argument('--dataset', default=None, help='Choose one of [Veri776, VERIWILD, Market1501, VehicleID]')
    parser.add_argument('--imgsize_x', default=None, type=int, help='width image')
    parser.add_argument('--imgsize_y', default=None, type=int, help='height image')
    parser.add_argument('--num_instances', default=None, type=int, help='Number of images belonging to an ID inside of batch, the numbers of IDs is batch_size/num_instances')
    parser.add_argument('--model_arch', default=None, help='Model Architecture')
    parser.add_argument('--softmax_loss', default=None, help='The loss used for classification')
    parser.add_argument('--metric_loss', default=None, help='The loss used as metric loss')
    parser.add_argument("--triplet_margin", default=None, type=float, help='With margin>0 uses normal triplet loss. If margin<=0 or None Soft Margin Triplet Loss is used instead!')
    parser.add_argument('--optimizer', default=None, help='Adam or SGD')
    parser.add_argument('--initial_lr', default=None, type=float, help='Initial learning rate after warm-up')
    parser.add_argument('--lambda_ce', default=None, type=float, help='multiplier of the classification loss')
    parser.add_argument('--lambda_triplet', default=None, type=float, help='multiplier of the metric loss')

    parser.add_argument('--parallel', default=None, help='Whether to used DataParallel for multi-gpu in one device')    
    parser.add_argument('--half_precision', default=None, help='Use of mixed precision') 
    parser.add_argument('--mean_losses', default=None, help='Use of mixed precision') 
    
    args = parser.parse_args()

    ### Load hyper parameters
    if args.config:
        with open(args.config, "r") as stream:
            data = yaml.safe_load(stream)
    else:
        with open("./config/config.yaml", "r") as stream:
            data = yaml.safe_load(stream)

    data['BATCH_SIZE'] = args.batch_size or data['BATCH_SIZE']
    data['p_hflip'] = args.hflip or data['p_hflip']
    data['y_length'] = args.imgsize_y or data['y_length']
    data['x_length'] = args.imgsize_x or data['x_length']
    data['p_rerase'] = args.randomerase or data['p_rerase']
    data['dataset'] = args.dataset or data['dataset']
    data['NUM_INSTANCES'] = args.num_instances or data['NUM_INSTANCES']
    data['model_arch'] = args.model_arch or data['model_arch']
    if args.triplet_margin is not None: data['triplet_margin'] = args.triplet_margin
    data['softmax_loss'] = args.softmax_loss or data['softmax_loss']
    data['metric_loss'] = args.metric_loss or data['metric_loss']
    data['optimizer'] = args.optimizer or data['optimizer']
    data['lr'] = args.initial_lr or data['lr']
    data['parallel'] = args.parallel or data['parallel']
    data['alpha_ce'] = args.lambda_ce or data['alpha_ce']
    data['beta_tri'] = args.lambda_triplet or data['beta_tri']
    data['backbone'] = args.backbone or data['backbone']
    data['half_precision'] = args.half_precision or data['half_precision']
    if args.mean_losses is not None: data['mean_losses'] = bool(args.mean_losses)

    alpha_ce= data['alpha_ce']
    beta_tri = data['beta_tri']

    # Set Seed for consistent and deterministic results
    set_seed(data['torch_seed'])

    print("\nConfig used:\n")
    print(data)
    print("\nEnd config\n")

    # TODO augmentations
    # Transformation augmentation
    test_transform = transforms.Compose([
                      transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                      transforms.Normalize(data['n_mean'], data['n_std'])
    ])
                  
    train_base_transform = transforms.Compose([
                    transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    transforms.Pad(10)])
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
    print(f'Selected device: {device}')

    # Get data loaders
    if data['dataset'] == 'Veri776':
        start_time = time.time()
        print('\nPreparing query dataset...')
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

        print('\nPreparing gallery dataset...')
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

        print('\nPreparing train dataset...')
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
        if 'fixed' in data['sampler'].lower():
            sampler = FixedOrderSampler(saved_batches_path=data['sampler_batches'],
                                        num_epochs=data['num_epochs']
                                        )
            data_train = DataLoader(data_train,
                                    batch_sampler=sampler,
                                    num_workers=data['num_workers_train'],
                                    collate_fn=train_collate_fn,
                                    persistent_workers=True)
        else:
            sampler = RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES'])
            data_train = DataLoader(data_train,
                                    sampler=sampler,
                                    num_workers=data['num_workers_train'],
                                    batch_size=data['BATCH_SIZE'],
                                    collate_fn=train_collate_fn,
                                    persistent_workers=True)

        loading_time = time.time() - start_time
        print(f'\nLoading datasets took {format_time(loading_time)}\n')

    clear_cache()

    # Create Model
    model = get_model(data, device)

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

    # If running with fp16 precision
    if data['half_precision']:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = False

    # Initiate a Logger with TensorBoard to store Scalars, Embeddings and the weights of the model
    logger = Logger(data)

    # Freeze backbone at warmupup epochs up to data['warmup_iters'] 
    if data['freeze_backbone_warmup']:
        for param in model.modelup2L3.parameters():
            param.requires_grad = False
        for param in model.modelL4.parameters():
            param.requires_grad = False

    if data['epoch_freeze_L1toL3'] > 0:
        # Freeze up to the penultimate layer    
        for param in model.modelup2L3.parameters():
            param.requires_grad = False
        print("\nFrozen backbone before branches!\n")
  
    # Training Loop
    for epoch in tqdm(range(data['num_epochs']), desc='Training', unit='epoch', bar_format='{l_bar}{bar:80}{r_bar}'):
        # Unfreeze backbone
        if epoch == data['warmup_iters'] - 1: 
            for param in model.modelup2L3.parameters():
                param.requires_grad = True
            for param in model.modelL4.parameters():
                param.requires_grad = True
     
        if epoch == data['epoch_freeze_L1toL3'] - 1:
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
            for param in model.modelup2L3.parameters():
                param.requires_grad = True
            print("\n\nUnfroze backbone before branches!")
        
        # Step schedule
        if epoch >= data['epoch_freeze_L1toL3'] - 1:
            # with warnings.catch_warnings(category=UserWarning, action="ignore"):            
            scheduler.step()

        # Train Loop
        time_since_start = time.time() - script_start_time
        print(f'\nTime since script start: {format_time(time_since_start)}')

        if 'fixed' in data['sampler'].lower():
            sampler.set_epoch(epoch)
        start_time = time.time()
        train_loss, ce_loss, triplet_loss, alpha_ce, beta_tri = train_epoch(model, device, data_train, ce_loss_fn, metric_loss_fn, optimizer, data, alpha_ce, beta_tri, logger, epoch, scheduler, scaler)
        epoch_time = time.time() - start_time
        print(f'\nTotal train epoch {epoch + 1} time: {format_time(epoch_time)}')

        clear_cache()

        # Evaluation
        if epoch == 0 or (epoch + 1) % data['validation_period'] == 0 or epoch >= data['num_epochs'] - 15:
            print('Start evaluation...')
            start_time = time.time()
            cmc, mAP = test_epoch(model, device, data_query, data_gallery, data['model_arch'], logger, epoch, remove_junk=True, scaler=scaler)
            evaluation_time = time.time() - start_time
            print(f'\nTotal evaluation time: {format_time(evaluation_time)}')
            print(f'\nEpoch {epoch + 1}/{data["num_epochs"]}: Train Loss {train_loss:.4f} | CrossEntropy Loss {ce_loss:.4f} | Triplet Loss {triplet_loss:.4f} | Test mAP {mAP:.4f} | CMC1 {cmc[0]:.4f} | CMC5 {cmc[4]:.4f}\n')
            logger.save_model(model)

    print(f"Best mAP: {np.max(logger.logscalars['Accuraccy/mAP'])}")
    print(f"Best CMC1: {np.max(logger.logscalars['Accuraccy/CMC1'])}")
    logger.save_log()   
