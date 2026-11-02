"""Numerical-difference (ND) loss of the counting branch (paper Sec. 4).

For the classes below ``uper_numer_idx`` (the numbers) the negative log-likelihood is weighted by
``1 + alpha * |argmax - target| ** gamma``; the remaining classes use the plain negative log-likelihood, of which the
``keep_ratio`` hardest samples are kept. With ``uper_numer_idx=0`` (configs/EarthVLCount.py) no class is weighted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalLoss(nn.Module):
    def __init__(self, uper_numer_idx, alpha=1.0, keep_ratio=1.0, gamma=0.5, min_keep=1, scale_factor=0.5):
        super(NumericalLoss, self).__init__()
        self.uper_numer_idx = uper_numer_idx
        self.alpha = alpha
        self.gamma = gamma
        self.min_keep = min_keep
        self.keep_ratio = keep_ratio
        self.scale_factor = scale_factor  # unused, kept for config compatibility
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: B x num_class logits
            targets: B class indices

        Returns:
            dict with ``numerical_loss`` (the loss to optimise) and statistics that are only logged: ``scaler``
            (cross-entropy / numerical_loss), ``cls_l``, ``count_l``, ``ohem_num``, ``ohem_pre_num``,
            ``number_pre_num``.
        """
        device = inputs.device
        pred_probs = inputs.softmax(dim=1)
        b, c = pred_probs.shape
        one_hot = F.one_hot(targets, num_classes=c)
        masked = targets < self.uper_numer_idx
        numer_losses = torch.tensor([0.], requires_grad=True).to(device)
        cls_losses = torch.tensor([0.], requires_grad=True).to(device)
        loss_dict = dict()

        if masked.float().sum() > 0:
            numer_gt, numer_probs, numer_onehot = targets[masked], pred_probs[masked], one_hot[masked]
            weights = self.alpha * (numer_probs.argmax(dim=1) - numer_gt).abs() ** self.gamma
            numer_losses = -(1 + weights) * torch.log((numer_probs * numer_onehot).sum(dim=1))

        if (~masked).float().sum() > 0:
            cls_one_hot, cls_probs = one_hot[~masked], pred_probs[~masked]
            cls_losses = - torch.log((cls_probs * cls_one_hot).sum(dim=1))
            # keep the hardest keep_ratio of the samples
            num_inst = cls_losses.numel()
            if int(self.keep_ratio * num_inst) >= self.min_keep:
                num_hns = int(self.keep_ratio * num_inst)
                cls_losses, _ = cls_losses.reshape(-1).topk(num_hns, -1)
                valid_num = torch.tensor([num_hns], requires_grad=False).to(device).float()
            else:
                valid_num = torch.tensor([self.min_keep], requires_grad=False).to(device).float()

        numer_size = numer_losses.numel() if masked.float().sum() > 0 else torch.tensor([0.], requires_grad=False).to(device)
        cls_size = valid_num if (~masked).float().sum() > 0 else torch.tensor([0.], requires_grad=False).to(device)

        numerical_loss = (cls_losses.sum() + numer_losses.sum()) / (numer_size + cls_size + 1e-9)
        loss_dict['numerical_loss'] = numerical_loss
        if b > 0:
            loss_dict['scaler'] = self.ce_loss(inputs, targets.long()) / numerical_loss
        else:  # no sample with a number in the batch: the cross-entropy of nothing is NaN
            loss_dict['scaler'] = torch.zeros_like(numerical_loss)
        loss_dict['cls_l'] = cls_losses.sum() / (cls_size + 1e-9)
        loss_dict['count_l'] = numer_losses.sum() / (numer_size + 1e-9)
        loss_dict['ohem_num'] = cls_size
        loss_dict['ohem_pre_num'] = (~masked).float().sum()
        loss_dict['number_pre_num'] = masked.float().sum()
        return loss_dict
