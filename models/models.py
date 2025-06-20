import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from torch import Tensor
# from torchsummary import summary
# import numpy as np
import copy
import warnings
from transformers import AutoModel
from utils import count_parameters


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MHA(nn.Module):
    def __init__(self, n_dims, heads=4):
        super(MHA, self).__init__()
        self.query = nn.Linear(n_dims, n_dims)
        self.key = nn.Linear(n_dims, n_dims)
        self.value = nn.Linear(n_dims, n_dims)
        self.mha = torch.nn.MultiheadAttention(n_dims, heads)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        out = self.mha(q, k, v)
        return out


## Transformer Block
##multi Head attetnion from BoTnet https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
class MHSA(nn.Module):
    def __init__(self, n_dims, width=16, height=16, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        ### , bias = False in conv2d
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size() # C // self.heads,
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, torch.div(C, self.heads, rounding_mode='floor'), -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

###also from https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py

class BottleneckTransformer(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, resolution=None, use_mlp = False):
        super(BottleneckTransformer, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList()
        self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
        if stride == 2:
            self.conv2.append(nn.AvgPool2d(2, 2))
        self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.use_MLP = use_mlp
        if use_mlp:
            self.LayerNorm = torch.nn.InstanceNorm2d(in_planes)
            self.MLP_torch = torchvision.ops.MLP(in_planes, [512, 2048])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.use_MLP:
            residual = out
            out = self.LayerNorm(out)
            out = out.permute(0,3,2,1)
            out = self.MLP_torch(out)
            out = out.permute(0,3,2,1)
            out = out + residual
            # out = F.relu(out)
        return out


# Defines the new fc layer and classification layer
# |--MLP--|--bn--|--relu--|--Linear--|
class ClassificationBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.0, relu=False, bnorm=True, linear=False, return_features = True, circle=False):
        super(ClassificationBlock, self).__init__()
        self.return_features = return_features
        self.circle = circle
        add_block = []
        if linear: ####MLP to reduce
            final_dim = linear
            add_block += [nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, final_dim)]
        else:
            final_dim = input_dim
        if bnorm:
            tmp_block = nn.BatchNorm1d(final_dim)
            tmp_block.bias.requires_grad_(False) 
            add_block += [tmp_block]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(final_dim, class_num, bias=False)] # 
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        if x.dim()==4:
            x = x.squeeze().squeeze()
        if x.dim()==1:
            x = x.unsqueeze(0)
        x = self.add_block(x)
        if self.return_features:
            f = x
            if self.circle:
                x = F.normalize(x)
                self.classifier[0].weight.data = F.normalize(self.classifier[0].weight, dim=1)
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class Conv_MHSA_2G(nn.Module):
    def __init__(self, c_in, c_out, resolution=[16,16], heads=4) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(c_in//2, c_out//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.MHSA_1 = MHSA(c_out//2, width=int(resolution[0]), height=int(resolution[1]), heads=heads)

    def forward(self,x):
        x_1 = self.conv2(x[:,:x.size(1)//2,:,:])
        x_2 = self.MHSA_1(x[:,x.size(1)//2:,:,:])

        x = torch.cat((x_1, x_2), dim=1)

        return x


class Conv_MHSA_4G(nn.Module):
    def __init__(self, c_in, c_out, resolution=[16, 16], heads=4) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(c_in // 2, c_out // 2, kernel_size=3, stride=1, padding=1, groups=2, bias=False)
        self.MHSA_1 = MHSA(c_out // 4, width=resolution[0], height=resolution[1], heads=heads)
        self.MHSA_2 = MHSA(c_out // 4, width=resolution[0], height=resolution[1], heads=heads)

    def forward(self,x):
        x_12 = self.conv2(x[:, :x.size(1) // 2, :, :])
        x_3 = self.MHSA_1(x[:, x.size(1) // 2:x.size(1) // 2 + x.size(1) // 4, :, :])
        x_4 = self.MHSA_2(x[:, x.size(1) // 2 + x.size(1) // 4:, :, :])
        x = torch.cat((x_12, x_3, x_4), dim=1)

        return x


class MHSA_2G(nn.Module):
    def __init__(self, c_out, resolution=[16,16], heads=4) -> None:
        super().__init__()
        self.MHSA_1 = MHSA(int(c_out//2), width=int(resolution[0]), height=int(resolution[1]), heads=heads)
        self.MHSA_2 = MHSA((c_out//2), width=int(resolution[0]), height=int(resolution[1]), heads=heads)

    def forward(self,x):
        x_ce = self.MHSA_1(x[:,:x.size(1)//2,:,:])
        x_t = self.MHSA_2(x[:,x.size(1)//2:,:,:])
        x = torch.cat((x_ce, x_t), dim=1)

        return x


class DinoV2Backbone(nn.Module):
    def __init__(self, model_name: str, n_groups: int):
        super().__init__()
        self.model = AutoModel.from_pretrained(f"facebook/{model_name}")
        self.n_groups = n_groups
    
    def convert_to_backbone_format(self, t: Tensor) -> Tensor:
        batch_size, n_patches, n_channels = t.shape
        side_length = int(torch.sqrt(torch.tensor(n_patches, dtype=float)).item())
        t_reshaped = t[:, 1:, :].reshape(batch_size, side_length, -1, n_channels).permute(0, 3, 1, 2)
        channels_per_group = 1024 / self.n_groups
        group_shift = int((n_channels - channels_per_group) // (self.n_groups - 1))
        t_grouped = torch.cat([t_reshaped[:, i * group_shift:i * group_shift + 256, :, :] for i in range(self.n_groups)], dim=1)
        return t_grouped

    def forward(self, images: Tensor) -> Tensor:
        outputs = self.model(pixel_values=images)
        patch_features = outputs.last_hidden_state  # Shape: (B, N, D)
        reshaped_patch_features = self.convert_to_backbone_format(patch_features)
        return reshaped_patch_features


class BaseBranches(nn.Module):
    def __init__(self, backbone="ibn", stride: int = 1, n_groups: int = 4):
        super(BaseBranches, self).__init__()
        new_backbone = False
        if backbone == 'r50':
            print(f'Backbone: resnet50')
            model_ft = models.resnet50()
        elif backbone == '101ibn':
            print(f'Backbone: resnet101_ibn_a')
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
        elif backbone == '34ibn':
            print(f'Backbone: resnet34_ibn_a')
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_a', pretrained=True)
        elif 'dinov2' in backbone:
            print(f'Backbone: {backbone}')
            new_backbone = True
            model_ft = DinoV2Backbone(backbone, n_groups=n_groups)
        else:
            print('Backbone: resnet50_ibn_a')
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)

        if not new_backbone:
            if stride == 1:
                model_ft.layer4[0].downsample[0].stride = (1,1)
                if backbone == "34ibn":
                    model_ft.layer4[0].conv1.stride = (1,1)
                else:
                    model_ft.layer4[0].conv2.stride = (1,1)

            self.model = torch.nn.Sequential(*(list(model_ft.children())[:-3]))
        else:
            self.model = model_ft

    def forward(self, x):
        x = self.model(x)
        return x


class MultiBranches(nn.Module):
    def __init__(self,
                 branches: list[str],
                 n_groups: int,
                 pretrain_ongroups: bool = True,
                 end_bot_g: bool  = False,
                 group_conv_mhsa: bool  = False,
                 group_conv_mhsa_2: bool  = False,
                 x2g: bool = False,
                 x4g: bool  = False
                 ):
        super(MultiBranches, self).__init__()

        model_ft: nn.Module = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        model_ft: nn.Sequential = model_ft.layer4
        self.x2g = x2g
        self.x4g = x4g
        if n_groups > 0:
            convlist = [k.split('.') for k, m in model_ft.named_modules(remove_duplicate=False) if isinstance(m, nn.Conv2d)]
            for item in convlist:
                if item[1] == "downsample":
                    m = model_ft[int(item[0])].get_submodule(item[1])[0]
                else:
                    m = model_ft[int(item[0])].get_submodule(item[1]) #'.'.join(
                weight = m.weight[:int(m.weight.size(0)), :int(m.weight.size(1)/n_groups), :,:]

                if end_bot_g and item[1]=="conv2":
                    setattr(model_ft[int(item[0])], item[1], MHSA_2G(int(512), int(512)))
                elif group_conv_mhsa and item[1]=="conv2":
                    setattr(model_ft[int(item[0])], item[1], Conv_MHSA_4G(int(512), int(512)))
                elif group_conv_mhsa_2 and item[1]=="conv2":
                    setattr(model_ft[int(item[0])], item[1], Conv_MHSA_2G(int(512), int(512)))
                else:
                    if item[1] == "downsample":
                        getattr(model_ft[int(item[0])], item[1])[0] = nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming)
                        if pretrain_ongroups:
                            getattr(model_ft[int(item[0])], item[1])[0].weight.data = weight
                    elif item[1] == "conv2":
                        setattr(model_ft[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=3, stride=1, padding=(1,1), groups=n_groups, bias=False).apply(weights_init_kaiming))
                        if pretrain_ongroups:
                            setattr(model_ft[int(item[0])].get_submodule(item[1]).weight, "data", weight)                        
                    else:
                        setattr(model_ft[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming))
                        if pretrain_ongroups:
                            setattr(model_ft[int(item[0])].get_submodule(item[1]).weight, "data", weight)

        self.model = nn.ModuleList()

        if len(branches) > 0:
            if branches[0] == "2x":
                self.model.append(model_ft)
                self.model.append(copy.deepcopy(model_ft))
            else:
                for item in branches:
                    if item =="R50":
                        self.model.append(copy.deepcopy(model_ft))
                    elif item == "BoT":
                        layer_0 = BottleneckTransformer(1024, 512, resolution=[16, 16], use_mlp = False)
                        layer_1 = BottleneckTransformer(2048, 512, resolution=[16, 16], use_mlp = False)
                        layer_2 = BottleneckTransformer(2048, 512, resolution=[16, 16], use_mlp = False)
                        self.model.append(nn.Sequential(layer_0, layer_1, layer_2))
                    else:
                        print("No valid architecture selected for branching by expansion!")
        else:
            self.model.append(model_ft)
        
        # print(self.model)

    def forward(self, x):
        output = []
        for cnt, branch in enumerate(self.model):
            if self.x2g and cnt > 0:
                aux = torch.cat((x[:,int(x.shape[1]/2):,:,:], x[:,:int(x.shape[1]/2),:,:]), dim=1)
                output.append(branch(aux))
            elif self.x4g and cnt > 0:
                aux = torch.cat((x[:,int(x.shape[1]/4):int(x.shape[1]/4*2),:,:], x[:, :int(x.shape[1]/4),:,:], x[:, int(x.shape[1]/4*3):,:,:], x[:, int(x.shape[1]/4*2):int(x.shape[1]/4*3),:,:]), dim=1)
                output.append(branch(aux))
            else:
                output.append(branch(x))
       
        return output


class FinalLayer(nn.Module):
    def __init__(self,
                 n_classes: int,
                 branches: list[str],
                 n_groups: int,
                 losses: str = "LBS",
                 droprate: float = 0,
                 linear_num: bool = False,
                 return_classification_features: bool = True,
                 circle_softmax: bool = False,
                 n_cams: int = 0,
                 n_views: int = 0,
                 LAI: bool = False,
                 x2g: bool = False,
                 x4g: bool = False,
                 ):
        super(FinalLayer, self).__init__()    
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.finalblocks = nn.ModuleList()
        self.withLAI = LAI
        self.LBS = losses == "LBS"
        if n_groups > 0:
            self.n_groups = n_groups
            for i in range(n_groups * (len(branches) + 1)):
                if self.LBS:
                    if i % 2 == 0:
                        # even branches are classification, uneven are with metric loss
                        self.finalblocks.append(ClassificationBlock(int(2048 / n_groups), n_classes, droprate, linear=linear_num, return_features=return_classification_features, circle=circle_softmax))
                    else:
                        bn= nn.BatchNorm1d(int(2048 / n_groups))
                        bn.bias.requires_grad_(False)  
                        bn.apply(weights_init_kaiming)
                        self.finalblocks.append(bn)
                else:
                    self.finalblocks.append(ClassificationBlock(int(2048 / n_groups), n_classes, droprate, linear=linear_num, return_features=return_classification_features, circle=circle_softmax))
        else:
            # not MBR_4G
            self.n_groups = 1
            for i in range(len(branches)):
                if self.LBS:
                    if i % 2==0:
                        self.finalblocks.append(ClassificationBlock(2048, n_classes, droprate, linear=linear_num, return_features = return_classification_features, circle=circle_softmax))
                    else:
                        bn= nn.BatchNorm1d(int(2048))
                        bn.bias.requires_grad_(False)  
                        bn.apply(weights_init_kaiming)
                        self.finalblocks.append(bn)
                else:
                    self.finalblocks.append(ClassificationBlock(2048, n_classes, droprate, linear=linear_num, return_features = return_classification_features, circle=circle_softmax))


    def forward(self, x, cam, view):
        # if len(x) != len(self.finalblocks):
        #     print("Something is wrong")
        aux_embs = []
        embeddings = []
        preds = []
        for i in range(len(x)):
            emb = self.avg_pool(x[i]).squeeze(dim=-1).squeeze(dim=-1)
            for j in range(self.n_groups):
                aux_emb = emb[:, int(2048 / self.n_groups * j):int(2048 / self.n_groups * (j + 1))]
                if self.LBS:
                    if (i + j) % 2 == 0:
                        pred, ff = self.finalblocks[i + j](aux_emb)
                        embeddings.append(ff)
                        preds.append(pred)
                    else:
                        ff = self.finalblocks[i + j](aux_emb)
                        aux_embs.append(aux_emb)
                        embeddings.append(ff)
                else:
                    aux_emb = emb[:, int(2048 / self.n_groups * j):int(2048 / self.n_groups * (j + 1))]
                    pred, ff = self.finalblocks[i + j](aux_emb)
                    aux_embs.append(aux_emb)
                    embeddings.append(ff)
                    preds.append(pred)
                    
        return preds, aux_embs, embeddings


class MBR_model(nn.Module):         
    def __init__(self,
                 branches: list[str],
                 n_groups: int,
                 n_classes: int,
                 backbone: str = "ibn",
                 losses: str = "LBS",
                 n_cams: int = 0,
                 n_views: int = 0,
                 droprate: float = 0,
                 return_f: bool = True,
                 pretrain_ongroups: bool = True,
                 LAI: bool = False,
                 linear_num: bool = False,
                 circle_softmax: bool = False,
                 end_bot_g: bool = False,
                 group_conv_mhsa: bool = False,
                 group_conv_mhsa_2: bool = False,
                 x2g: bool = False,
                 x4g: bool = False,
                 ):
        super(MBR_model, self).__init__()  

        self.modelup2L3 = BaseBranches(backbone=backbone,
                                       n_groups=n_groups)
        self.modelL4 = MultiBranches(branches=branches,
                                     n_groups=n_groups,
                                     pretrain_ongroups=pretrain_ongroups,
                                     end_bot_g=end_bot_g,
                                     group_conv_mhsa=group_conv_mhsa,
                                     group_conv_mhsa_2=group_conv_mhsa_2,
                                     x2g=x2g,
                                     x4g=x4g)
        self.finalblock = FinalLayer(branches=branches,
                                     n_classes=n_classes,
                                     n_groups=n_groups,
                                     losses=losses,
                                     n_cams=n_cams,
                                     n_views=n_views,
                                     droprate=droprate,
                                     return_classification_features=return_f,
                                     LAI=LAI,
                                     linear_num=linear_num,
                                     circle_softmax=circle_softmax,
                                     x2g=x2g,
                                     x4g=x4g)
        
    def forward(self, images, cams, views):
        mix = self.modelup2L3(images)
        output = self.modelL4(mix)
        preds, embs, embeddings = self.finalblock(output, cams, views)

        return preds, embs, embeddings, output
    
    def montecarlo_predict(self, images, cams, views, iters: int = 100):
        self.modelup2L3.train()
        mc_res = []
        for _ in range(iters):
            mix = self.modelup2L3(images)
            output = self.modelL4(mix)
            preds, embs, embeddings = self.finalblock(output, cams, views)
            mc_res.append((torch.stack(preds), torch.stack(embs), torch.stack(embeddings), torch.stack(output)))
        
        averaged = tuple(
            torch.stack(tensors).mean(dim=0)
            for tensors in zip(*mc_res)
        )
        return averaged
