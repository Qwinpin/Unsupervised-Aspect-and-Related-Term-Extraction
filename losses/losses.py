import numpy as np
import torch
import torch.functional as F
from torch import nn


def l2norm(x):
    numerator = x
    denominator = torch.sum(x.pow(2), -1, keepdims=True)
    denominator = torch.sqrt(F.relu(denominator) + 1e-10)

    return numerator / denominator


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = l2norm(anchor)
        positive = l2norm(positive)
        negative = l2norm(negative)

        positive_2 = torch.unsqueeze(positive, -2)
        positive_2 = positive_2.repeat(1, negative.size()[1], 1)

        f_a_p = (positive - anchor).pow(2).sum(1)
        f_a_n = (positive_2 - negative).pow(2).sum(1)
        f_a_n = torch.mean(f_a_n, -1)

        hard_triplets = torch.where(f_a_n < f_a_p)[0]

        f_a_p = f_a_p[hard_triplets]
        f_a_n = f_a_n[hard_triplets]
        error = F.relu(f_a_p - f_a_n + self.margin)

        return torch.sum(error, -1, keepdims=True)


class MaxMarginLoss(nn.Module):
    def __init__(self, margin):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = l2norm(anchor)
        positive = l2norm(positive)
        negative = l2norm(negative)

        steps = negative.size()[1]
        f_p = torch.sum(positive * anchor, axis=-1, keepdims=True)
        f_p = f_p.repeat(1, steps)

        positive = torch.unsqueeze(positive, -2)
        positive = positive.repeat(1, steps, 1)

        f_n = torch.sum(positive * negative, -1)

        loss = torch.sum(F.relu(1.0 - f_p + f_n), -1, keepdims=True)
        return loss


class DualLoss(nn.Module):
    """
    Increase margin between first two attentioned sentences and decrease
    margin between attentioned sentences and corresponding aspect
    """
    def __init__(self, margin, top_N, dist_koef1=0.05, dist_koef2=1.0):
        super(DualLoss, self).__init__()
        self.margin = margin
        self.top_N = top_N
        self.dist_koef1 = dist_koef1
        self.dist_koef2 = dist_koef2

    def forward(self, aw, attentioned_sent, aspect_probability, aspects):
        attentioned_sent = l2norm(attentioned_sent)

        tmp_top = np.array([([(k, j[0]) for j in sorted([(i, asp) for i, asp in enumerate(sample)], key=lambda x: -x[1])[:self.top_N]]) for k, sample in enumerate(aspect_probability.cpu().detach().numpy())])
        top = np.concatenate([tmp_top[:, 0], tmp_top[:, 1]])

        attentioned_sent_selected = attentioned_sent[top[:, 0], top[:, 1]]
        attentioned_sent_selected_grouped = [attentioned_sent_selected[i: i + self.top_N].unsqueeze(0) for i in range(attentioned_sent_selected.size()[0])[::self.top_N]]
        attentioned_sent_selected_grouped = torch.cat(attentioned_sent_selected_grouped)

        aw_selected = aw[top[:, 1]]
        aw_selected_grouped = [aw_selected[i: i + self.top_N].unsqueeze(0) for i in range(aw_selected.size()[0])[::self.top_N]]
        aw_selected_grouped = torch.cat(aw_selected_grouped)

        distance_positive = self.dist_koef1 * (attentioned_sent_selected_grouped - aw_selected_grouped).pow(2).sum(-1).mean(1)
        distance_negative = self.dist_koef2 * (attentioned_sent_selected_grouped[:, 0] - attentioned_sent_selected_grouped[:, 1]).pow(2).sum(1)

        loss = torch.sum(F.relu(1.0 + distance_positive - distance_negative), -1, keepdims=True)
        return loss


class TailReduction(nn.Module):
    """
    Increase probability of first N aspects, decrease summary probability of others to zero
    """
    def __init__(self, koef=1.0):
        super(TailReduction, self).__init__()
        self.koef = koef

    def forward(self, x, head_len=3):
        x = torch.sort(x, 1)[0]

        head = x[:, -head_len:]
        x = x[:, :-head_len]
        loss = self.koef * torch.sum(x, 1)
        loss_head = torch.max(head, 1)[0] - torch.min(head, 1)[0]
        return (loss + loss_head).sum()


def ortho_reg(weight_matrix, koef, device='cpu'):
    # orthogonal regularization for aspect embedding matrix
    w_n = (weight_matrix / (1e-10 + torch.sqrt(torch.sum(weight_matrix.pow(2), axis=-1, keepdims=True))).view(weight_matrix.size()[0], 1)).float()

    reg = torch.sum((torch.matmul(w_n, w_n.transpose(1, 0)) - torch.eye(w_n.size()[0]).to(device)).pow(2)).float()

    return koef * (reg - 0.1).abs()


class OrtogonalLoss(nn.Module):
    def __init__(self, koef, device):
        super(OrtogonalLoss, self).__init__()
        self.koef = koef
        self.device = device

    def forward(self, weights):
        return ortho_reg(weights, self.koef, self.device)
