import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import math

class ProportionalScalePad:
    """
    Scale the image proportionally so the largest dimension matches target size,
    then pad the smaller dimension with black pixels to reach target size.
    If image is smaller in both dimensions, scales up before padding.
    
    Args:
        target_x (int): Target width size
        target_y (int): Target height size
        interpolation (InterpolationMode): Torchvision interpolation mode
    """
    def __init__(self, target_width, target_height, interpolation=transforms.InterpolationMode.BILINEAR):
        self.target_width = target_width
        self.target_height = target_height
        self.interpolation = interpolation
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be transformed
            
        Returns:
            PIL Image or Tensor: Transformed image
        """
        # Get current dimensions
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape[-3], img.shape[-2], img.shape[-1]
        else:
            w, h = img.size

        # if w < self.target_x and h < self.target_y:  
        #     # Calculate scaling factor
        scale_w = self.target_width / w
        scale_h = self.target_height / h
        scale = min(scale_w, scale_h)
        img = F.resize(img, [math.ceil(h * scale), math.ceil(w * scale)], interpolation=self.interpolation)

        new_h, new_w = img.shape[1:]
        
        # Calculate padding
        pad_left = (self.target_width - new_w) // 2
        pad_right = self.target_width - new_w - pad_left
        pad_top = (self.target_height - new_h) // 2
        pad_bottom = self.target_height - new_h - pad_top
        
        # Pad the image
        img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
        
        assert img.shape[1:] == (self.target_height, self.target_width)
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}(target_width={self.target_width}, target_height={self.target_height}, interpolation={self.interpolation})"