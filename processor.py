from turtle import update
import torch
import torch.nn.functional as F
from models.models import MBR_model
from tqdm import tqdm
import numpy as np
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
        # ### 2B hybrid No LBS   
        # if 'Hybrid_2B' == data['model_arch']:
        #     model = MBR_model(n_classes=data['n_classes'], branches=["R50", "BoT"], n_groups=0, losses="Classical", LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # ### 2B R50 No LBS
        # if 'R50_2B' == data['model_arch']:
        #     model = MBR_model(n_classes=data['n_classes'], branches=["R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # ### 2B R50 LBS
        # if data['model_arch'] == 'MBR_R50_2B':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["R50", "R50"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # ### Baseline with BoT
        # if data['model_arch'] == 'BoT_baseline':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["BoT"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # ### 2B BoT LBS
        # if data['model_arch'] == 'MBR_BOT_2B':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["BoT", "BoT"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # ### MBR-4B (4B hybrid LBS)
        # if data['model_arch'] == 'MBR_4B':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["R50", "R50", "BoT", "BoT"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        
        # ### 4B hybdrid No LBS
        # if data['model_arch'] == 'Hybrid_4B':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["R50", "R50", "BoT", "BoT"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # ### 4B R50 No LBS
        # if data['model_arch'] == 'R50_4B':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["R50", "R50", "R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])    

        # if data['model_arch'] == 'MBR_R50_4B':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["R50", "R50", "R50", "R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])


        # ### 4G hybrid No LBS
        # if data['model_arch'] =='Hybrid_4G':
        #     model = MBR_model(n_classes=data['n_classes'], branches=[], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # if data['model_arch'] =='MBR_2x2G':    
        #     model = MBR_model(n_classes=data['n_classes'], branches=['2x'], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x2g=True, group_conv_mhsa_2=True) 

        # if data['model_arch'] =='MBR_R50_2x2G':  
        #     model = MBR_model(n_classes=data['n_classes'], branches=['2x'], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x2g=True)  

        # ### 2G BoT LBS
        #     model = MBR_model(n_classes=data['n_classes'], branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], end_bot_g=True)

        # ### 2G R50 LBS
        # if data['model_arch'] =='MBR_R50_2G':
        #     model = MBR_model(n_classes=data['n_classes'], branches=[], losses="LBS", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # ### 2G Hybrid No LBS
        # if data['model_arch'] =='Hybrid_2G':
        #     model = MBR_model(n_classes=data['n_classes'], branches=[], losses="Classical", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], group_conv_mhsa_2=True)

        # ### 2G R50 No LBS
        # if data['model_arch'] =='R50_2G':
        #     model = MBR_model(n_classes=data['n_classes'], branches=[], losses="Classical", n_groups=2, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # ### 4G R50 No LBS
        # if data['model_arch'] =='R50_4G':
        #     model = MBR_model(n_classes=data['n_classes'], branches=[], losses="Classical", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

        # ### 4G only R50 with LBS
        # if data['model_arch'] =='MBR_R50_4G':
        #     model = MBR_model(n_classes=data['n_classes'], branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], group_conv_mhsa_2=True)
        
        # if data['model_arch'] =='MBR_R50_2x4G':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["2x"], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x4g=True)

        # if data['model_arch'] =='MBR_2x4G':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["2x"], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'], x4g=True, group_conv_mhsa=True)

        # if data['model_arch'] == 'Baseline':
        #     model = MBR_model(n_classes=data['n_classes'], branches=["R50"], losses="Classical", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
        model = None

    return model.to(device)


def train_epoch(model, device, dataloader, loss_fn, triplet_loss, optimizer, data, alpha_ce, beta_triplet, logger, epoch, scheduler=None, scaler=False):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = []
    ce_loss_log = []
    triplet_loss_log = []

    gamma_ce = data['gamma_ce']
    gamma_t = data['gamma_t']
    model_arch = data['model_arch']

    loss_ce_log =      tqdm(total=0, position=1, bar_format='{desc}', leave=True)
    loss_triplet_log = tqdm(total=0, position=2, bar_format='{desc}', leave=True)
    loss_log =         tqdm(total=0, position=3, bar_format='{desc}', leave=True)

    n_images = 0
    acc_v = 0
    stepcount = 0
    for batch_id, (batch_images, batch_labels, batch_cams, batch_views, batch_indices, batch_workers) in enumerate(tqdm(dataloader, position=4, desc=f'Epoch {epoch + 1}', bar_format='{l_bar}{bar:20}{r_bar}', unit='batch', leave=True)): 
        # Move tensor to the proper device
        loss_ce = 0
        loss_triplet = 0
        optimizer.zero_grad()
        batch_images = batch_images.to(device=device, non_blocking=True)
        batch_labels = batch_labels.to(device=device, non_blocking=True)
        # with open('log.txt', 'a') as f:
        #     print(f'{batch_id}:', batch_indices, file=f)
        # torch.save(batch_indices.type(torch.int32), f'batch_logs/epoch_{epoch:03d}/batch_{batch_id:05d}_worker_{torch.unique(batch_workers).item()}.pt')
        # with open('log_batches_no_preload.txt', 'a') as f:
        #     f.write(f'Worker {torch.unique(batch_workers).item()}:\n{'\n'.join(map(str, batch_indices.tolist()))}')
        # if batch_id == 0:
        #     print(f'BATCH {batch_id}: last image (idx {batch_indices[-1]}):', batch_images[-1])
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds, embs, _, _ = model(batch_images, batch_cams, batch_views)
                loss = 0
                # Losses 
                if type(preds) != list:
                    preds = [preds]
                    embs = [embs]
                for i, item in enumerate(preds):
                    if i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch:
                        loss_ce += alpha_ce * loss_fn(item, batch_labels)
                    else:
                        loss_ce += gamma_ce * loss_fn(item, batch_labels)
                for i, item in enumerate(embs):
                    if i % 2 == 0 or "aseline" in model_arch or "R50" in model_arch:
                        loss_triplet += beta_triplet * triplet_loss(item, batch_labels)
                    else:
                        loss_triplet += gamma_t * triplet_loss(item, batch_labels)

                if data['mean_losses']:
                    loss = loss_ce/len(preds) + loss_triplet/len(embs)
                else:
                    loss = loss_ce + loss_triplet
        else:
            preds, embs, _, _ = model(batch_images, batch_cams, batch_views)

            loss = 0
            # Losses 
            if type(preds) != list:
                preds = [preds]
                embs = [embs]
            for i, item in enumerate(preds):
                if i % 2==0 or "aseline" in model_arch or "R50" in model_arch:
                    loss_ce += alpha_ce * loss_fn(item, batch_labels)
                else:
                    loss_ce += gamma_ce * loss_fn(item, batch_labels)
            for i, item in enumerate(embs):
                if i % 2==0 or "aseline" in model_arch or "R50" in model_arch:
                    loss_triplet += beta_triplet * triplet_loss(item, batch_labels)
                else:
                    loss_triplet += gamma_t * triplet_loss(item, batch_labels)

            if data['mean_losses']:
                loss = loss_ce/len(preds) + loss_triplet/len(embs)
            else:
                loss = loss_ce + loss_triplet

        # Training Acurracy
        for prediction in preds:
            acc_v += torch.sum(torch.argmax(prediction, dim=1) == batch_labels)
            n_images += prediction.size(0)
        stepcount += 1
    
        ### backward prop and optimizer step
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

        train_loss.append(loss.detach().cpu().numpy())
        ce_loss_log.append(loss_ce.detach().cpu().numpy())
        triplet_loss_log.append(loss_triplet.detach().cpu().numpy())

    logger.write_scalars({"Loss/train_total": np.mean(train_loss), 
                          "Loss/train_crossentropy": np.mean(ce_loss_log),
                          "Loss/train_triplet": np.mean(triplet_loss_log),
                          # "Loss/ce_loss_weight": alpha_ce,
                          # "Loss/triplet_loss_weight": beta_triplet,
                          # "lr/learning_rate": get_lr(optimizer),
                          "Loss/AccuracyTrain": (acc_v / n_images).cpu().numpy()},
                          epoch * len(dataloader) + stepcount,
                          write_epoch=True
                          )

    print('\n\n\n\n', end='')
    print(f'Train Accuracy: {float((acc_v / n_images).cpu().item())}\n')
    return np.mean(train_loss), np.mean(ce_loss_log), np.mean(triplet_loss_log), alpha_ce, beta_triplet


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
