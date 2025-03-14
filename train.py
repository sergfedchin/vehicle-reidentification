import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
from data.triplet_sampler import *
from loss.losses import TripletLossFastReID
from lr_scheduler.sche_optim import make_optimizer, make_warmup_scheduler
import argparse
import torch.multiprocessing
import yaml
import os
from tensorboard_log import Logger
from processor import get_model, train_epoch, test_epoch
from pprint import pprint


torch.multiprocessing.set_sharing_strategy('file_system')


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False


def set_module_param_attributes(module: nn.Module, attribute: str, value):
    for param in module.parameters():
        setattr(param, attribute, value)


def get_device(data) -> torch.device:
    # Force a GPU
    if not data['parallel']:  
        if data['gpu']:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(data['gpu'])
    # Check if the GPU is available and select
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    if device == torch.device('cuda'):
        print('-' * 20, 'GPU info', '-' * 20)
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"{num_gpus} GPU Available.")
            for i in range(num_gpus):
                gpu_info = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {gpu_info.name}")
                print(f"  Total Memory:          {gpu_info.total_memory / 1024**2} MB")
                print(f"  Multi Processor Count: {gpu_info.multi_processor_count}")
                print(f"  Compute Capability:    {gpu_info.major}.{gpu_info.minor}")
        torch.cuda.empty_cache()
    return device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ReID model trainer')
    parser.add_argument('--config', default=None, help='Config Path')
    parser.add_argument('--dataset', default=None, help='Choose one of [Veri776, VERIWILD, Market1501, VehicleID]')
    parser.add_argument('--model_arch', default=None, help='Model Architecture')
 
    args = parser.parse_args()

    # Load hyper parameters
    if args.config:
        with open(args.config, "r") as stream:
            data = yaml.safe_load(stream)
    else:
        print('No config file specified. Proceeding with the default config "config/config.yaml".')
        with open("./config/config.yaml", "r") as stream:
            data = yaml.safe_load(stream)

    data['dataset'] = args.dataset or data['dataset']
    data['model_arch'] = args.model_arch or data['model_arch']

    alpha_ce = data['alpha_ce']
    beta_tri = data['beta_tri']

    # Set Seed for consistent and deterministic results
    set_seeds(data['torch_seed'])
    print("\nConfig used:")
    pprint(data)
    print('\n')

    # Transformation augmentation
    test_transform = transforms.Compose([
                    transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    transforms.Normalize(data['n_mean'], data['n_std']),

    ])                  
    train_transform = transforms.Compose([
                    transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    transforms.Pad(10),
                    transforms.RandomCrop((data['y_length'], data['x_length'])),
                    transforms.RandomHorizontalFlip(p=data['p_hflip']),
                    transforms.Normalize(data['n_mean'], data['n_std']),
                    transforms.RandomErasing(p=data['p_rerase'], value=0),
    ])        

    if data['dataset'] == 'Veri776':
        data_q = CustomDataSet4Veri776_withviewpont(data['query_list_file'],   data['query_dir'], viewpoints=data['viewpoints'], is_train=False, transform=test_transform)
        data_g = CustomDataSet4Veri776_withviewpont(data['gallery_list_file'], data['test_dir'], viewpoints=data['viewpoints'], is_train=False, transform=test_transform)
        if data["LAI"]:
            data_train = CustomDataSet4Veri776_withviewpont(data['train_list_file'], data['train_dir'], viewpoints=data['viewpoints'], is_train=True, transform=train_transform)
        else:
            data_train = CustomDataSet4Veri776(data['train_list_file'], data['train_dir'], is_train=True, transform=train_transform)
        data_q     = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_test'])
        data_g     = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_test'])
        data_train = DataLoader(data_train,
                                sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']),
                                num_workers=data['num_workers_train'],
                                batch_size = data['BATCH_SIZE'],
                                collate_fn=train_collate_fn,
                                pin_memory=True)
 
    # Create Model
    device = get_device(data)
    model = get_model(data, device)

    if data['parallel']:
        model = torch.nn.DataParallel(model, device_ids=torch.arange(torch.cuda.device_count()))
        print("\nParallel activated!\nDo not use this with LBS!\nIt may result in weird behaviour sometimes.")

    # Losses
    loss_fn = nn.CrossEntropyLoss(label_smoothing=data['label_smoothing'])
    metric_loss = TripletLossFastReID(data['triplet_margin'], norm_feat=data['triplet_norm'], hard_mining=data['hard_mining'])
    
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
    scaler = torch.amp.GradScaler('cuda') if data['half_precision'] else False  

    # Initiate a Logger with TensorBoard to store Scalars, Embeddings and the weights of the model
    logger = Logger(data)

    # Freeze backbone at warmup epochs up to data['warmup_iters'] 
    if data['freeze_backbone_warmup']:
        set_module_param_attributes(model.modelup2L3, 'requires_grad', False)
        set_module_param_attributes(model.modelL4, 'requires_grad', False)
    if data['epoch_freeze_L1toL3'] > 0:
        # Freeze up to the penultimate layer    
        set_module_param_attributes(model.modelup2L3, 'requires_grad', False)
        print("\nFroze Backbone before branches!")

    # Training Loop
    for epoch in trange(data['num_epochs']):
        # Unfreeze backbone
        if epoch == data['warmup_iters'] - 1: 
            set_module_param_attributes(model.modelup2L3, 'requires_grad', True)
            set_module_param_attributes(model.modelL4, 'requires_grad', True)
     
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
            set_module_param_attributes(model.modelup2L3, 'requires_grad', True)
            print("\nUnfrozen Backbone before branches!")
        
        # step schedule
        if epoch >= data['epoch_freeze_L1toL3'] - 1:              
            scheduler.step()    
        # Train Loop
        train_loss, c_loss, t_loss, alpha_ce, beta_tri = train_epoch(model=model,
                                                                     device=device,
                                                                     dataloader=data_train,
                                                                     loss_fn=loss_fn,
                                                                     triplet_loss_fn=metric_loss,
                                                                     optimizer=optimizer, 
                                                                     data=data,
                                                                     alpha_ce=alpha_ce,
                                                                     beta_tri=beta_tri,
                                                                     logger=logger,
                                                                     epoch=epoch,
                                                                     scheduler=scheduler,
                                                                     scaler=scaler)
        # Evaluation
        if epoch % data['validation_period'] == 0 or epoch >= data['num_epochs'] - 15:
            cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], logger, epoch, remove_junk=True, scaler=scaler)
            print('\n EPOCH {}/{} \t train loss {} \t Classification loss {} \t Triplet loss {} \t mAP {} \t CMC1 {} \t CMC5 {}'.format(epoch + 1, data['num_epochs'], train_loss, c_loss, t_loss,mAP, cmc[0], cmc[4]))
            logger.save_model(model)

    print("Best mAP: ", np.max(logger.logscalars['Accuraccy/mAP']))
    print("Best CMC1: ", np.max(logger.logscalars['Accuraccy/CMC1']))
    logger.save_log()   
