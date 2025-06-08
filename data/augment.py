import torch


def mixup_data(x, y, cams, views, device, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(device)
    else:
        lam = torch.tensor(1.0, device=device)

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    cams_a, cams_b = cams, cams[index]
    views_a, views_b = views, views[index]
    
    return mixed_x, y_a, y_b, cams_a, cams_b, views_a, views_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_triplet_loss(triplet_loss, emb, y_a, y_b, lam):
    loss_a = triplet_loss(emb, y_a)
    loss_b = triplet_loss(emb, y_b)
    return lam * loss_a + (1 - lam) * loss_b