import torch
import torch.nn.functional as F
from models.models import MBR_model
from tqdm import tqdm
from data.augment import *

from metrics.eval_reid import eval_func


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model: torch.nn.Module): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(data, device):
    # 4G hybryd with LBS  MBR-4G
    if data['model_arch'] == 'MBR_4G':
        model = MBR_model(branches=[],
                          n_groups=4,
                          n_classes=data['n_classes'],
                          backbone=data['backbone'],
                          losses="LBS",
                          n_cams=data['n_cams'],
                          n_views=data['n_views'],
                          LAI=data['LAI'],
                          )
    if data['model_arch'] != 'MBR_4G':
        model = None
    return model.to(device)


def train_epoch(model, device, dataloader, loss_fn, triplet_loss, optimizer, data, alpha_ce, beta_triplet, logger, epoch, scheduler=None, scaler=False):
    # Set train mode for both the encoder and the decoder
    model.train()
    losses = []
    losses_ce = []
    losses_triplet = []

    gamma_ce = data['gamma_ce']
    gamma_t = data['gamma_t']
    model_arch = data['model_arch']
    use_mixup = data.get('mixup', False)
    tqdm.write(f'Use MixUp: {use_mixup}')

    loss_ce_log =      tqdm(total=0, position=1, bar_format='{desc}', leave=True)
    loss_triplet_log = tqdm(total=0, position=2, bar_format='{desc}', leave=True)
    loss_log =         tqdm(total=0, position=3, bar_format='{desc}', leave=True)

    n_images = 0
    acc_v = 0
    stepcount = 0

    for batch_id, (batch_images, batch_labels, batch_cams, batch_views, batch_indices, batch_workers) in enumerate(tqdm(dataloader, position=4, desc=f'Epoch {epoch + 1}', bar_format='{l_bar}{bar:20}{r_bar}', unit='batch', leave=True)): 
        # Move tensor to the proper device
        batch_images = batch_images.to(device=device, non_blocking=True)
        batch_labels = batch_labels.to(device=device, non_blocking=True)
        batch_cams = batch_cams.to(device=device, non_blocking=True)
        batch_views = batch_views.to(device=device, non_blocking=True)
        
        # Apply MixUp if enabled
        if use_mixup:
            batch_images, labels_a, labels_b, cams_a, cams_b, views_a, views_b, lam = mixup_data(
                batch_images, batch_labels, batch_cams, batch_views, device=device, alpha=0.4
            )
        else:
            lam = torch.tensor(1.0, device=device)
            labels_a, labels_b = batch_labels, batch_labels
            cams_a, cams_b = batch_cams, batch_cams
            views_a, views_b = batch_views, batch_views

        loss_ce = torch.tensor(0.0, device=device)
        loss_triplet = torch.tensor(0.0, device=device)
        optimizer.zero_grad()
        
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds, embs, _, _ = model(batch_images, cams_a, views_a)
                loss = torch.tensor(0.0, device=device)
                
                if type(preds) != list:
                    preds = [preds]
                    embs = [embs]
                
                for i, item in enumerate(preds):
                    current_alpha = alpha_ce if (i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch) else gamma_ce
                    if use_mixup:
                        loss_ce += current_alpha * mixup_criterion(loss_fn, item, labels_a, labels_b, lam)
                    else:
                        loss_ce += current_alpha * loss_fn(item, batch_labels)
                
                for i, item in enumerate(embs):
                    current_beta = beta_triplet if (i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch) else gamma_t
                    if use_mixup:
                        loss_triplet += current_beta * mixup_triplet_loss(triplet_loss, item, labels_a, labels_b, lam)
                    else:
                        loss_triplet += current_beta * triplet_loss(item, batch_labels)

                if data['mean_losses']:
                    loss = loss_ce/len(preds) + loss_triplet/len(embs)
                else:
                    loss = loss_ce + loss_triplet
        else:
            preds, embs, _, _ = model(batch_images, cams_a, views_a)
            loss = torch.tensor(0.0, device=device)
            
            if type(preds) != list:
                preds = [preds]
                embs = [embs]
            
            for i, item in enumerate(preds):
                current_alpha = alpha_ce if (i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch) else gamma_ce
                if use_mixup:
                    loss_ce += current_alpha * mixup_criterion(loss_fn, item, labels_a, labels_b, lam)
                else:
                    loss_ce += current_alpha * loss_fn(item, batch_labels)
            
            for i, item in enumerate(embs):
                current_beta = beta_triplet if (i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch) else gamma_t
                if use_mixup:
                    loss_triplet += current_beta * mixup_triplet_loss(triplet_loss, item, labels_a, labels_b, lam)
                else:
                    loss_triplet += current_beta * triplet_loss(item, batch_labels)

            if data['mean_losses']:
                loss = loss_ce/len(preds) + loss_triplet/len(embs)
            else:
                loss = loss_ce + loss_triplet

        # Training Accuracy
        temp_labels = labels_a if (use_mixup and lam > 0.5) else batch_labels
        for prediction in preds:
            acc_v += torch.sum(torch.argmax(prediction, dim=1) == temp_labels)
            n_images += prediction.size(0)
        stepcount += 1
    
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_ce_log.set_description_str(     f'CrossEntropy Loss: {loss_ce.data:.4f}')
        loss_triplet_log.set_description_str(f'Triplet Loss:      {loss_triplet.data:.4f}')
        loss_log.set_description_str(        f'Total Train loss:  {loss.data:.4f}')

        losses_ce.append(loss_ce.detach().cpu())
        losses_triplet.append(loss_triplet.detach().cpu())
        losses.append(loss.detach().cpu())

    loss_mean_ce = sum(losses_ce) / len(losses_ce)
    loss_mean_triplet = sum(losses_triplet) / len(losses_triplet)
    loss_mean = sum(losses) / len(losses)

    logger.write_scalars({"Loss/train_total": loss_mean, 
                          "Loss/train_crossentropy": loss_mean_ce,
                          "Loss/train_triplet": loss_mean_triplet,
                          "Loss/AccuracyTrain": (acc_v / n_images).cpu().item()},
                          epoch * len(dataloader) + stepcount,
                          write_epoch=True
                          )

    print('\n\n\n\n', end='')
    print(f'Train Accuracy: {float((acc_v / n_images).cpu().item())}\n')
    return loss_mean, loss_mean_ce, loss_mean_triplet, alpha_ce, beta_triplet


def test_epoch(model, device, dataloader_q, dataloader_g, model_arch, writer, epoch, remove_junk=True, scaler=False):
    model.eval()

    qf = []
    gf = []
    q_camids = []
    g_camids = []
    q_vids = []
    g_vids = []
    q_images = []
    g_images =  []

    with torch.no_grad():
        for image, q_id, cam_id, view_id in tqdm(dataloader_q, desc='Embedding query', bar_format='{l_bar}{bar:20}{r_bar}', leave=True, unit='batch'):
            image = image.to(device=device, non_blocking=True)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = model(image, cam_id, view_id)
            else:
                _, _, ffs, _ = model(image, cam_id, view_id)
          
            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            qf.append(torch.cat(end_vec, 1))

            q_vids.append(q_id)
            q_camids.append(cam_id)

            if epoch == 119:  
                q_images.append(F.interpolate(image, (64, 64)).cpu())

        # TensorBoard emmbeddings for projector visualization
        if epoch == 119:    
            writer.write_embeddings(torch.cat(qf).cpu(), torch.cat(q_vids).cpu(), torch.cat(q_images)/2 + 0.5, 120, tag='Query embeddings')

        del q_images

        for image, q_id, cam_id, view_id in tqdm(dataloader_g, desc='Embedding gallery', bar_format='{l_bar}{bar:20}{r_bar}', leave=True, unit='batch'):
            image = image.to(device=device, non_blocking=True)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = model(image, cam_id, view_id)
            else:
                    _, _, ffs, _ = model(image, cam_id, view_id)

            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            gf.append(torch.cat(end_vec, 1))
            g_vids.append(q_id)
            g_camids.append(cam_id)
        del g_images

    qf = torch.cat(qf, dim=0)
    gf = torch.cat(gf, dim=0)
    m, n = qf.shape[0], gf.shape[0]   
    distmat =  torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(),beta=1, alpha=-2)
    distmat = torch.sqrt(distmat).cpu().numpy()

    q_camids = torch.cat(q_camids, dim=0).cpu().numpy()
    g_camids = torch.cat(g_camids, dim=0).cpu().numpy()
    q_vids = torch.cat(q_vids, dim=0).cpu().numpy()
    g_vids = torch.cat(g_vids, dim=0).cpu().numpy()   
    
    del qf, gf
    
    cmc, mAP = eval_func(distmat, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk)

    writer.write_scalars({"Accuraccy/CMC1": cmc[0], "Accuraccy/mAP": mAP}, epoch)

    return cmc, mAP
