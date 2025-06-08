from __future__ import annotations
from collections import defaultdict
from tqdm import tqdm, trange

import numpy as np
import copy
import random
import os
import os.path as osp
import pickle

import torch
import torchvision
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


def train_collate_fn(batch):
    imgs, pids, camids, viewids, indices, workers = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    indices = torch.tensor(indices, dtype=torch.int64)
    workers = torch.tensor(workers, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids, indices, workers


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index in range(len(self.data_source.data_info)):
            pid = self.data_source.get_class(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length  


class FixedOrderSampler(Sampler):
    def __init__(self, saved_batches_path, num_epochs: int = 130):
        """
        Args:
            saved_batches_path (str): Path to .pt file with shape [epochs, batches_per_epoch, batch_size]
            num_epochs (int): Total epochs in saved file
        """
        # Load the pre-saved batch indices
        self.current_epoch = 0

        with open(saved_batches_path, 'rb') as f:
            self.epochs_batches = pickle.load(f)

        assert len(self.epochs_batches) >= num_epochs

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        """Yields batches of current epoch"""
        for batch in self.epochs_batches[self.current_epoch]:
            yield batch
        
    def __len__(self):
        """Number of batches in epoch"""
        return len(self.epochs_batches[self.current_epoch])


mp.set_start_method('spawn', force=True)


def read_image_and_transform(image_path: str,
                             transform: torchvision.transforms.Compose | bool,
                             use_fp16: bool = False,
                             device: torch.device = torch.device('cpu'),
                             ) -> torch.Tensor:
    image = torchvision.io.read_image(image_path)
    # if image_path == '/home/serg_fedchn/Homework/6_semester/НИР/object-reidentification/dataset/veri_images/0370_c009_00064640_0.jpg':
    #     print(f'INSIDE read_image_and_transform after just read:', image)
    if transform:
        image = transform((image.type(torch.FloatTensor)) / 255.0)
    if use_fp16:
        image = image.half()
    image = image.to(device)
    # if image_path == '/home/serg_fedchn/Homework/6_semester/НИР/object-reidentification/dataset/veri_images/0370_c009_00064640_0.jpg':
    #     print(f'INSIDE read_image_and_transform before moving to device:', image)
    return image


class LoaderDataset(Dataset):
    """Simpel dataset class for loading from disk"""
    def __init__(self, paths, transform, use_fp16, preload_mask):
        self.paths = paths
        self.transform = transform
        self.use_fp16 = use_fp16
        self.preload_mask = preload_mask

        self.empty_image = torch.zeros_like(self.transform(torch.zeros(3, 256, 256)))
        if self.use_fp16:
            self.empty_image = self.empty_image.half()
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        if self.preload_mask[idx]:
            # chosen to load
            image = read_image_and_transform(self.paths[idx], self.transform, self.use_fp16)
            if idx == 10679:
                print(f'LOADING IMAGE 10679: read_image_and_transform({self.paths[idx]}, {self.transform}, {self.use_fp16})')
                print(f'LOADED:', image)
        else:
            image = self.empty_image
        return idx, image


def preload_to_device_parallel(image_paths: list[str],
                               transform: torchvision.transforms.Compose | bool,
                               num_workers: int,
                               use_fp16: bool,
                               preload_device: torch.device,
                               preload_rate: float,
                               preload_batch_size: int
                               ) -> list[torch.Tensor | None]:
    """Parallel loading with CUDA handling. Returns list of `Tensors`
    with images (and `Nones` when `preload_rate` < 1)"""
    imgs = [None] * len(image_paths)
    if preload_rate <= 0:
        return imgs
    if preload_rate >= 1:
        preload_mask = torch.ones(len(image_paths)).type(torch.bool)
    else:
        preload_mask = torch.rand(len(image_paths)) < preload_rate
    
    loader_dataset = LoaderDataset(image_paths, transform, use_fp16, preload_mask)
    dataloader = DataLoader(
        loader_dataset,
        batch_size=preload_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # was TRUE for server!!!!!!!!!!!!!!!!! TODO
        persistent_workers=False
    )

    # Initialize CUDA context in main process
    torch.zeros(1).to(preload_device)
    # Then move to CUDA in main process
    for batch in tqdm(dataloader, desc=f'Preloading {int(preload_rate * 100)}% to {str(preload_device).upper()}', unit='batch'):
        for index, image in zip(*batch):
            if preload_mask[index]:
                image = image.to(preload_device)
                # if index == 10679:
                #     print(f'SAVED AT INDEX 10679:', image)
                imgs[index] = image
    del dataloader
    torch.cuda.empty_cache()
    return imgs


class CUDADatasetVeri776(Dataset):
    def __init__(self,
                 image_list: str,
                 images_dir: str,
                 is_train: bool = True,
                 base_transform: torchvision.transforms.Compose | bool = False,
                 random_transform: torchvision.transforms.Compose | bool = False,
                 device: str | torch.device = 'cuda',
                 use_fp16: bool = True,
                 preload_rate: float = 1,
                 preload_num_workers: int = 4,
                 preload_batch_size: int = 32,
                 ):
        self.root_dir = images_dir
        self.base_transform = base_transform
        self.random_transform = random_transform
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        
        # Read and parse image list
        with open(image_list) as f:
            lines = f.readlines()
        
        # parsing file names
        self.names, self.labels, self.cams = tuple(map(list, zip(*[(line, *((lambda x: (int(x[0]), int(x[1][1:])))(line.split('_')[:2]))) for line in map(lambda s: s.strip(), lines)])))
        
        # Remap labels for training
        if is_train:
            unique_labels = sorted(set(self.labels))
            label_map = {id: pid for pid, id in enumerate(unique_labels)}
            self.labels = [label_map[id] for id in self.labels]

        self.paths = list(map(lambda x: osp.join(self.root_dir, x), self.names))

        # Preload images
        # self.imgs = preload_to_device_parallel(image_paths=self.paths,
        #                                        transform=base_transform,
        #                                        num_workers=preload_num_workers,
        #                                        use_fp16=use_fp16,
        #                                        preload_device=device,
        #                                        preload_rate=preload_rate,
        #                                        preload_batch_size=preload_batch_size)
        self.imgs = [None] * len(self.names)
        self.data_info = self.names
        self.num_preloaded = 0
        self.preload_rate = preload_rate

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # print(f'REQUESTED IDX', idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        worker_id = torch.utils.data.get_worker_info().id
        vehicle_id = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)
        cam_id = torch.tensor(self.cams[idx], dtype=torch.long, device=self.device)
        # image = self.imgs[idx]
        # if image is None:
        #     image = read_image_and_transform(self.paths[idx], self.transform, self.use_fp16, self.device)
        #     if idx == 10679:
        #         print(f'LOADING IMAGE 10679: read_image_and_transform({self.paths[idx]}, {self.transform}, {self.use_fp16}, {self.device})')
        #         print(f'LOADED:', image)
        if (image := self.imgs[idx]) is None:
            image = read_image_and_transform(self.paths[idx], self.base_transform, self.use_fp16, self.device)
            if (self.num_preloaded + 1) / len(self.names) <= self.preload_rate:
                self.imgs[idx] = image
                self.num_preloaded += 1
        return self.random_transform(image), vehicle_id, cam_id, 0, idx, worker_id


class CUDADatasetVeri776Viewpoints(Dataset):
    def __init__(self,
                 image_list: str,
                 images_dir: str,
                 viewpoints: str,
                 is_train: bool = True,
                 transform: torchvision.transforms.Compose | bool = False,
                 device: str | torch.device = 'cuda',
                 use_fp16: bool = True,
                 preload_rate: float = 1,
                 preload_num_workers: int = 4,
                 preload_batch_size: int = 32,
                 ):
        self.root_dir = images_dir
        self.transform = transform
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        with open(image_list) as f:
            lines = list(map(lambda s: s.strip(), f.readlines()))

        self.names, self.labels, self.cams = tuple(map(list, zip(*[(line, *((lambda x: (int(x[0]), int(x[1][1:])))(line.split('_')[:2]))) for line in map(lambda s: s.strip(), lines)])))

        with open(viewpoints, 'r') as f:
            viewpoints = {line[0]: list(map(int, line[1:])) for line in map(lambda s: s.strip().split(), f.readlines())}
        n_missing_views = 0
        self.view = []
        for line in lines:
            view = viewpoints.get(line, '')
            if not view:
                n_missing_views += 1
                continue
            self.view.append(view[-1] if view else 0)

        if is_train == True:
            unique_labels = sorted(set(self.labels))
            label_map = {id: pid for pid, id in enumerate(unique_labels)}
            self.labels = [label_map[id] for id in self.labels]

        self.paths = list(map(lambda x: osp.join(self.root_dir, x), self.names))
        # Preload images
        self.imgs = preload_to_device_parallel(image_paths=self.paths,
                                               transform=transform,
                                               num_workers=preload_num_workers,
                                               use_fp16=use_fp16,
                                               preload_device=device,
                                               preload_rate=preload_rate,
                                               preload_batch_size=preload_batch_size)
        self.data_info = self.names

        print(f'Missed viewpoint for {n_missing_views}/{len(self.view)} images for {"train" if is_train else "evaluation"}!')

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        vehicle_id = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)
        cam_id = torch.tensor(self.cams[idx], dtype=torch.long, device=self.device)
        view_id = torch.tensor(self.view[idx], dtype=torch.long, device=self.device)
        image = self.imgs[idx]
        if image is None:
            image = read_image_and_transform(self.paths[idx], self.transform, self.use_fp16, self.device)
        return image, vehicle_id, cam_id, view_id    
