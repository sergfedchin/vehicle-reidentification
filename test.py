import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from metrics.eval_reid import *
from data.triplet_sampler import *
from typing import OrderedDict
from processor import get_model
import torch.multiprocessing
import os
import os.path as osp
import yaml
from utils import re_ranking
import time


def normalize_batch(batch, maximo=None, minimo = None):
    if maximo != None:
        return (batch - minimo.unsqueeze(-1).unsqueeze(-1)) / (maximo.unsqueeze(-1).unsqueeze(-1) - minimo.unsqueeze(-1).unsqueeze(-1))
    else:
        return (batch - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)) / (torch.amax(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1) - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False


def test_epoch(model, device, dataloader_q, dataloader_g, model_arch, remove_junk=True, scaler=None, re_rank=False):
    model.eval()
    # re_escala = torchvision.transforms.Resize((256,256), antialias=True)
    qf = []
    gf = []
    q_camids = []
    g_camids = []
    q_vids = []
    g_vids = []
    q_images = []
    g_images =  []
    count_imgs = 0
    # blend_ratio =0.3
    with torch.no_grad():
        for image, q_id, cam_id, view_id  in tqdm(dataloader_q, desc='Embedding query', bar_format='{l_bar}{bar:20}{r_bar}', leave=True, unit='batch'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, activations = model(image, cam_id, view_id)
            else:
                _, _, ffs, activations = model(image, cam_id, view_id)
                    
            count_imgs += activations[0].shape[0]
            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            qf.append(torch.cat(end_vec, 1))
            q_vids.append(q_id)
            q_camids.append(cam_id)

        del q_images
        count_imgs = 0
        for image, g_id, cam_id, view_id in tqdm(dataloader_g, desc='Embedding gallery', bar_format='{l_bar}{bar:20}{r_bar}', leave=True, unit='batch'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, activations = model(image, cam_id, view_id)
            else:
                _, _, ffs, activations = model(image, cam_id, view_id)

            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            gf.append(torch.cat(end_vec, 1))
            g_vids.append(g_id)
            g_camids.append(cam_id)

            count_imgs += activations[0].shape[0]

        del g_images

    qf = torch.cat(qf, dim=0)
    gf = torch.cat(gf, dim=0)

    m, n = qf.shape[0], gf.shape[0]   
    if re_rank:
        distmat = re_ranking(qf, gf, k1=80, k2=16, lambda_value=0.3)
    else:
        distmat =  torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(),beta=1, alpha=-2)
        distmat = torch.sqrt(distmat).cpu().numpy()

    q_camids = torch.cat(q_camids, dim=0).cpu().numpy()
    g_camids = torch.cat(g_camids, dim=0).cpu().numpy()
    q_vids = torch.cat(q_vids, dim=0).cpu().numpy()
    g_vids = torch.cat(g_vids, dim=0).cpu().numpy()   
    del qf, gf

    
    cmc, mAP = eval_func(distmat, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk, num_visual_samples=7, output_dir="vis_results")
    print(f'mAP = {mAP}, CMC1 = {cmc[0]}, CMC5 = {cmc[4]}')

    return cmc, mAP


if __name__ == "__main__":
    ### Just to ensure VehicleID 10-fold validation randomness is not random to compare different models training
    set_seed(0)
    parser = argparse.ArgumentParser(description='Reid train')

    parser.add_argument('--batch_size', default=None, type=int, help='an integer for the accumulator')
    parser.add_argument('--dataset', default=None, help='Choose one of[Veri776, VERIWILD]')
    parser.add_argument('--model_arch', default=None, help='Model Architecture')
    parser.add_argument('--path_weights', default=None, help="Path to *.pth/*.pt loading weights file")
    parser.add_argument('--re_rank', action="store_true", help="Re-Rank")
    args = parser.parse_args()

    path_weights = args.path_weights
    log_folder_path = osp.dirname(args.path_weights)

    with open(osp.join(log_folder_path, "config.yaml"), "r") as stream:
        data = yaml.safe_load(stream)

    data['BATCH_SIZE'] = args.batch_size or data['BATCH_SIZE']
    data['dataset'] = args.dataset or data['dataset']
    data['model_arch'] = args.model_arch or data['model_arch']


    test_transform = transforms.Compose([
                    transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    transforms.Normalize(data['n_mean'], data['n_std']),
    ])                  

    if data['half_precision']:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = False

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

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
        loading_time = time.time() - start_time
        print(f'\nLoading datasets took {int(loading_time // 60)}m {int(loading_time % 60)}s\n')

    print('Loading model...')
    start_time = time.time()
    model = get_model(data, torch.device("cpu"))

    try:
        model.load_state_dict(torch.load(path_weights, map_location='cpu')) 
    except RuntimeError:
        ### nn.Parallel adds "module." to the dict names. Although like said nn.Parallel can incur in weird results in some cases 
        tmp = torch.load(path_weights, map_location='cpu')
        tmp = OrderedDict((k.replace("module.", ""), v) for k, v in tmp.items())
        model.load_state_dict(tmp)

    model = model.to(device)
    model.eval()

    loading_time = time.time() - start_time
    print(f'Loading model took {int(loading_time // 60)}m {int(loading_time % 60)}s\n')

    mean = False
    l2 = True

    cmc, mAP = test_epoch(model, device, data_query, data_gallery, data['model_arch'], remove_junk=True, scaler=scaler, re_rank=args.re_rank)

    print(f'\nEvaluation results: mAP = {mAP:.4f} | CMC1 = {cmc[0]:.4f} | CMC5 = {cmc[4]:.4f}')
    with open(osp.join(log_folder_path, f'result_map_l2_{l2}_mean_{mean}.npy'), 'wb') as f:
        np.save(f, mAP)
    with open(osp.join(log_folder_path + f'result_cmc_l2_{l2}_mean_{mean}.npy'), 'wb') as f:
        np.save(f, cmc)

    print('Weights: ', path_weights)
