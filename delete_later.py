from processor import get_model
import argparse
import torch
from torchvision import transforms
import yaml
import os
import os.path as osp
from data.triplet_sampler import read_image_and_transform
from transformers import AutoImageProcessor, AutoModel
import time

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert_to_backbone_format(t: torch.Tensor, n_groups: int):
    batch_size, n_pathes, n_channels = t.shape
    side_length = int(torch.sqrt(torch.tensor(n_pathes)).item())
    t_reshaped = t[:, 1:, :].reshape(batch_size, side_length, -1, n_channels).permute(0, 3, 1, 2)
    channels_per_group = 1024 / n_groups
    group_shift = int((n_channels - channels_per_group) // (n_groups - 1))
    t_grouped = torch.cat([t_reshaped[:, i * group_shift:i * group_shift + 256, :, :] for i in range(n_groups)], dim=1)
    return t_grouped


backbone = 'dinov2-small'
model = AutoModel.from_pretrained(f"facebook/{backbone}").to(device='cuda')

parser = argparse.ArgumentParser(description='ReID model trainer')
parser.add_argument('--config', default=None, help='Config Path')
args = parser.parse_args()

with open(args.config, "r") as stream:
    data = yaml.safe_load(stream)

dataset_path = 'dataset/veri_images'
images_paths = list(map(lambda x: osp.join(dataset_path, x), os.listdir(dataset_path)[:48]))

transform = transforms.Compose([
    # transforms.Resize((252, 252), antialias=True),
    transforms.Resize((256, 256), antialias=True),
])

images = torch.stack([read_image_and_transform(image_path, transform=transform, use_fp16=True, device='cuda') for image_path in images_paths])  / 255.0
print('Images batch:', images.shape)
with torch.amp.autocast(device_type='cuda'):
    outputs = model(pixel_values=images)
patch_features: torch.Tensor = outputs.last_hidden_state  # Shape: (B, N, D)
reshaped_patch_features = convert_to_backbone_format(patch_features, n_groups=4)
print("Patch tokens shape:", reshaped_patch_features.shape)

MBR_4G = get_model(data, 'cuda')
print(f"MBR_4G: {count_parameters(MBR_4G) / 1e6:.1f}M params")
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    MBR_4G_backbone_output = MBR_4G.modelup2L3(images)
print("Backbone IBN output shape:", MBR_4G_backbone_output.shape)
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    MBR_4G_output = MBR_4G(images, torch.zeros(5), torch.zeros(5))
print('MBR_4G output shape:', torch.cat(MBR_4G_output[2], 1).shape)

output = MBR_4G.modelL4(reshaped_patch_features)
MBR_4G_output = MBR_4G.finalblock(output, torch.zeros(5), torch.zeros(5))
print(f'Modified MBR_4G shape:', torch.cat(MBR_4G_output[2], 1).shape)