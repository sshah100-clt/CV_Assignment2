"""Self-supervised objectives and a kNN monitor of representation quality.

Each objective is a module that owns its own head (projector or rotation
classifier) and maps (encoder, views) -> (loss, accuracy of its pretext task).
Only the encoder is kept after pretraining; the head is discarded.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """SimCLR (Chen et al., 2020). Two random views of every image go through
    the encoder and a small MLP projector. With N images there are 2N
    embeddings; each must pick out the other view of the same image among the
    2N - 1 others (NT-Xent: cross-entropy over cosine similarities divided by
    a temperature). The other images in the batch are the negatives, so the
    encoder must keep what distinguishes one image from another while
    ignoring what the augmentations change (crop position, scale, brightness,
    contrast, blur). The loss is computed on the projector output; the
    encoder output before the projector is the representation that is kept."""

    def __init__(self, feat_dim, cfg):
        super().__init__()
        s = cfg['ssl']
        self.temperature = s['temperature']
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, s['proj_hidden']), nn.BatchNorm1d(s['proj_hidden']), nn.ReLU(inplace=True),
            nn.Linear(s['proj_hidden'], s['proj_dim']),
        )

    def forward(self, encoder, views):
        a, b = views
        z = F.normalize(self.projector(encoder(torch.cat([a, b]))).float(), dim=1)
        n = len(a)
        sim = z @ z.T / self.temperature
        sim.fill_diagonal_(float('-inf'))  # an embedding is not its own positive
        target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
        loss = F.cross_entropy(sim, target)
        acc = (sim.argmax(1) == target).float().mean()
        return loss, acc


class RotationPrediction(nn.Module):
    """RotNet (Gidaris et al., 2018). Every image is shown at 0, 90, 180 and 270
    degrees and the network predicts which rotation it sees. Scenes have a
    canonical orientation (sky above, floor below), so solving the task
    requires recognising sky, ground, walls and objects. A much simpler
    objective than SimCLR: no negatives, no projector, one view."""

    def __init__(self, feat_dim, cfg):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, 4)

    def forward(self, encoder, views):
        (x,) = views
        rotated = torch.cat([torch.rot90(x, k, dims=(2, 3)) for k in range(4)])
        target = torch.arange(4, device=x.device).repeat_interleave(len(x))
        logits = self.classifier(encoder(rotated)).float()
        return F.cross_entropy(logits, target), (logits.argmax(1) == target).float().mean()


def build_objective(cfg, feat_dim):
    return {'simclr': SimCLR, 'rotation': RotationPrediction}[cfg['ssl']['method']](feat_dim, cfg)


@torch.no_grad()
def embed(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for images, batch_labels in loader:
        feats.append(F.normalize(encoder(images.to(device)), dim=1).cpu())
        labels.append(batch_labels)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_accuracy(encoder, bank_loader, query_loader, device, k=20, temperature=0.1):
    """Weighted kNN classifier on frozen features (Wu et al., 2018): each
    validation image is classified by its k most similar training images
    (cosine similarity), each voting with weight exp(similarity / T).
    Needs no training, so it tracks representation quality during
    pretraining. Only training-fold labels are used for the votes."""
    bank, bank_labels = embed(encoder, bank_loader, device)
    query, query_labels = embed(encoder, query_loader, device)
    sim, idx = (query @ bank.T).topk(k, dim=1)
    votes = torch.zeros(len(query), int(bank_labels.max()) + 1)
    votes.scatter_add_(1, bank_labels[idx], (sim / temperature).exp())
    return (votes.argmax(1) == query_labels).float().mean().item()
