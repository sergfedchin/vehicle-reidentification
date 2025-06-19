import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import argparse
import random
import time
import yaml
import gc
from typing import Any, Iterable
from tqdm import tqdm


def count_parameters(model: nn.Module): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_modules_params_property(modules: Iterable[nn.Module], property: str, value):
    for module in modules:
        for param in module.parameters():
            param.__setattr__(property, value)


def format_time(t: int) -> str:
    return f'{f"{int(t // 3600)}h " if t >= 3600 else ""}'\
           f'{f"{int(t // 60) % 60}m " if t >= 60 else ""}'\
           f'{f"{int(t % 60)}" if t >= 10 else f"{t:.1f}"}s'


def clear_cache():
    tqdm.write('Emptying cache...', end=' ')
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    cache_emptying_time = time.time() - start_time
    tqdm.write(f'Took {format_time(cache_emptying_time)}\n')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark = False


def normalize_batch(batch, maximo=None, minimo = None):
    if maximo != None:
        return (batch - minimo.unsqueeze(-1).unsqueeze(-1)) / (maximo.unsqueeze(-1).unsqueeze(-1) - minimo.unsqueeze(-1).unsqueeze(-1))
    else:
        return (batch - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)) / (torch.amax(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1) - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1))


def get_train_data() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description='ReID model trainer')
    parser.add_argument('--config',         default=None, help='Config Path')
    parser.add_argument('--batch_size',     default=None, type=int, help='Batch size')
    parser.add_argument('--backbone',       default=None, help='Model Backbone')
    parser.add_argument('--hflip',          default=None, type=float, help='Probabilty for horizontal flip')
    parser.add_argument('--randomerase',    default=None, type=float,  help='Probabilty for random erasing')
    parser.add_argument('--dataset',        default=None, help='Choose one of [Veri776, VERIWILD, Market1501, VehicleID]')
    parser.add_argument('--imgsize_x',      default=None, type=int, help='width image')
    parser.add_argument('--imgsize_y',      default=None, type=int, help='height image')
    parser.add_argument('--num_instances',  default=None, type=int, help='Number of images belonging to an ID inside of batch, the numbers of IDs is batch_size/num_instances')
    parser.add_argument('--model_arch',     default=None, help='Model Architecture')
    parser.add_argument('--softmax_loss',   default=None, help='The loss used for classification')
    parser.add_argument('--metric_loss',    default=None, help='The loss used as metric loss')
    parser.add_argument("--triplet_margin", default=None, type=float, help='With margin>0 uses normal triplet loss. If margin<=0 or None Soft Margin Triplet Loss is used instead!')
    parser.add_argument('--optimizer',      default=None, help='Adam or SGD')
    parser.add_argument('--initial_lr',     default=None, type=float, help='Initial learning rate after warm-up')
    parser.add_argument('--lambda_ce',      default=None, type=float, help='multiplier of the classification loss')
    parser.add_argument('--lambda_triplet', default=None, type=float, help='multiplier of the metric loss')

    parser.add_argument('--parallel',       default=None, help='Use of DataParallel for multi-gpu in one device')    
    parser.add_argument('--half_precision', default=None, help='Use of mixed precision') 
    parser.add_argument('--mean_losses',    default=None, help='Use of mixed precision') 
    parser.add_argument('--mixup',          action="store_true", help='Use of MixUp augmentation') 
    
    args = parser.parse_args()

    # Load config
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
    data['mixup'] = args.mixup or data.get('mixup', False)
    if args.mean_losses is not None:
        data['mean_losses'] = bool(args.mean_losses)
    return data


def img_is_color(img: Tensor):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True
    return False


def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=False, num_cols=2, figsize=(20, 10), title_fontsize=30, number = None, directory2save = './Results/'):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    [axi.set_axis_off() for axi in axes.ravel()]
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)
        # list_axes[i].axis('off')

    fig.tight_layout()

    #_ = plt.show()
    plt.savefig(directory2save + str(number) + '.png', pad_inches=0.0, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close('all')
    del fig
    gc.collect()

def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        tqdm.write('Using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    tqdm.write('Starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
          
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)