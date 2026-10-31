"""Training and prediction loops."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def predict(model, loader, device):
    """Class probabilities and labels for all images, in loader order."""
    model.eval()
    probs, labels = [], []
    for images, batch_labels in loader:
        probs.append(model(images.to(device)).softmax(1).cpu())
        labels.append(batch_labels)
    return torch.cat(probs), torch.cat(labels)


def validate(model, loader, device):
    """Cross-entropy, accuracy and probabilities on a labelled loader."""
    probs, labels = predict(model, loader, device)
    loss = F.nll_loss(probs.clamp_min(1e-12).log(), labels).item()
    acc = (probs.argmax(1) == labels).float().mean().item()
    return loss, acc, probs, labels


def build_optimizer(params, lr, weight_decay):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def classifier_param_groups(model, cfg):
    """The classification head and the encoder are separate parameter groups,
    so a pretrained encoder can get a smaller learning rate, or be frozen
    (linear probe)."""
    o = cfg['optim']
    groups = [{'params': list(model.head.parameters()), 'lr': o['lr']}]
    if cfg['model']['freeze_encoder']:
        model.encoder.requires_grad_(False)
    else:
        groups.append({'params': list(model.encoder.parameters()), 'lr': o['lr'] * o['encoder_lr_mult']})
    return groups


def build_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch):
    """Learning-rate factor per step: linear warmup, then cosine decay."""
    total = epochs * steps_per_epoch
    warmup = warmup_epochs * steps_per_epoch

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, objective=None):
    """One pass over the labelled training data; returns mean classification
    loss, accuracy and (joint training only) mean SSL loss. With an SSL
    `objective`, each batch is [supervised view, SSL views...] and the loss is
    cross-entropy + ssl.joint_weight x SSL loss, both through the same encoder."""
    if cfg['model']['freeze_encoder']:
        model.eval()   # keeps the pretrained BatchNorm statistics fixed
        model.head.train()
    else:
        model.train()

    total_loss = total_ssl = correct = seen = 0
    for images, labels in loader:
        labels = labels.to(device)
        if objective is not None:
            images, *views = [v.to(device) for v in images]
        else:
            images = images.to(device)
        logits = model(images)
        ce = F.cross_entropy(logits, labels)
        loss = ce
        if objective is not None:
            ssl_loss, _ = objective(model.encoder, views)
            loss = ce + cfg['ssl']['joint_weight'] * ssl_loss
            total_ssl += ssl_loss.item() * len(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        correct += (logits.argmax(1) == labels).sum().item()
        total_loss += ce.item() * len(images)
        seen += len(images)
    return total_loss / seen, correct / seen, total_ssl / seen


def pretrain_one_epoch(encoder, objective, loader, optimizer, scheduler, device):
    """One pass of self-supervised pretraining; labels are never read.
    Returns mean SSL loss and mean pretext-task accuracy."""
    encoder.train()
    objective.train()
    total_loss = total_acc = steps = 0
    for views, _ in loader:
        # bfloat16 on CUDA halves activation memory (512 images per step); the
        # losses are computed in float32 (see ssl.py).
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss, acc = objective(encoder, [v.to(device) for v in views])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        total_acc += acc.item()
        steps += 1
    return total_loss / steps, total_acc / steps
