# Vehicle Re-Identification on VeRi776 Dataset

## Training

Prior to the training, <ins>all</ins> VeRi776 images (including `train`, `test` and `query`) should be loaded to `dataset/veri_images`. Files containing names of images in splits and viewpoints are located in `dataset/VeRi`.

Run the training process:
```shell
python3 train.py --config ./config/config_Veri776_MBR_4G.yaml
```
The results (model weights and metrics) are saved to `logs/Veri776/MBR_4G`.