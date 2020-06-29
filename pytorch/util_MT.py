import torch
import torch.nn.functional as F
from torch import nn

class EMAWeightOptimizer (object):
    def __init__(self, target_net, source_net, alpha=0.999):
        self.target_net = target_net
        self.source_net = source_net
        self.ema_alpha = alpha
        self.target_params = target_net.state_dict().values()
        self.source_params = source_net.state_dict().values()

        # print(self.target_params['fc.weight'])'fc.bias'

        for i, (tgt_p, src_p) in enumerate(zip(self.target_params, self.source_params)):
            tgt_p=src_p

        target_keys = set(target_net.state_dict().keys())
        source_keys = set(source_net.state_dict().keys())
        if target_keys != source_keys:
            raise ValueError('Source and target networks do not have the same state dict keys; do they have different architectures?')


    def step(self):
        one_minus_alpha = 1.0 - self.ema_alpha
        # for tgt_p, src_p in zip(self.target_params, self.source_params):
        #     tgt_p.mul_(self.ema_alpha)
        #     tgt_p.add_(src_p * one_minus_alpha)
        for i, (tgt_p, src_p) in enumerate(zip(self.target_params, self.source_params)):
            # if i == 638 or i == 639:
            #     tgt_p.mul_(self.ema_alpha)
            #     tgt_p.add_(src_p * one_minus_alpha)
            # else:
            #     tgt_p = src_p
            tgt_p.mul_(self.ema_alpha)
            tgt_p.add_(src_p * one_minus_alpha)

def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def compute_aug_loss(stu_out, tea_out,n_classes,confidence_thresh=0.65):
    # Augmentation loss
    conf_tea = torch.max(tea_out, 1)[0]
    unsup_mask = conf_mask = (conf_tea > confidence_thresh).float()
    unsup_mask_count = conf_mask_count = conf_mask.sum()

    d_aug_loss = stu_out - tea_out
    aug_loss = d_aug_loss * d_aug_loss

    # Class balance scaling

    n_samples = unsup_mask.sum()
    avg_pred = n_samples / float(n_classes)
    bal_scale = avg_pred / torch.clamp(tea_out.sum(dim=0), min=1.0)
    cls_bal_scale_range = 0.0
    if cls_bal_scale_range != 0.0:
        bal_scale = torch.clamp(bal_scale, min=1.0/cls_bal_scale_range, max=cls_bal_scale_range)
    bal_scale = bal_scale.detach()
    aug_loss = aug_loss * bal_scale[None, :]
    aug_loss = aug_loss.mean(dim=1)
    unsup_loss = (aug_loss * unsup_mask).mean()
    # Class balance loss
    cls_balance = 0.05

    if cls_balance > 0.0:
        # Compute per-sample average predicated probability
        # Average over samples to get average class prediction
        avg_cls_prob = stu_out.mean(dim=0)
        # Compute loss
        equalise_cls_loss = robust_binary_crossentropy(avg_cls_prob, float(1.0 / n_classes))
        equalise_cls_loss = equalise_cls_loss.mean() * n_classes
        equalise_cls_loss = equalise_cls_loss * unsup_mask.mean(dim=0)
        unsup_loss += equalise_cls_loss * cls_balance

    return unsup_loss

def compute_aug_loss2(stu_out, tea_out,n_classes,confidence_thresh=0.65):
    # Augmentation loss
    conf_tea = torch.max(tea_out, 1)[0]
    unsup_mask = conf_mask = (conf_tea > confidence_thresh).float()
    unsup_mask_count = conf_mask_count = conf_mask.sum()

    d_aug_loss = stu_out - tea_out
    aug_loss = d_aug_loss * d_aug_loss
    # aug_loss = - (tea_out * torch.log(stu_out))

    # Class balance scaling
    n_samples = unsup_mask.sum()

    aug_loss = aug_loss * n_samples.detach()
    aug_loss = aug_loss.mean(dim=1)
    unsup_loss = (aug_loss * unsup_mask).mean()

    return unsup_loss

def compute_cbloss(stu_out, n_classes, cls_balance=0.05):
    # Compute per-sample average predicated probability
    # Average over samples to get average class prediction
    avg_cls_prob = stu_out.mean(dim=0)
    # Compute loss
    equalise_cls_loss = robust_binary_crossentropy(avg_cls_prob, float(1.0 / n_classes))
    equalise_cls_loss = equalise_cls_loss.mean() * n_classes
    return equalise_cls_loss * cls_balance

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class VAT(nn.Module):
    def __init__(self, model):
        super(VAT, self).__init__()
        self.n_power = 1
        self.XI = 1e-6
        self.model = model
        self.epsilon = 3.5

    def forward(self, X, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = torch.randn_like(x, device='cuda')

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            _, logit_m = self.model(x + d)
            dist = self.kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit_p = logit.detach()
        _, logit_m = self.model(x + r_vadv)
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss

class CosineMarginLoss(nn.Module):
    def __init__(self, embed_dim, num_classes, m=0.35, s=64):
        super(CosineMarginLoss, self).__init__()
        self.w = nn.Parameter(torch.randn(31, num_classes))
        self.num_classes = num_classes
        self.m = m
        self.s = s

    def forward(self, x, labels):
        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        w_norm = self.w / torch.norm(self.w, dim=0, keepdim=True)
        xw_norm = torch.matmul(x_norm, w_norm)

        label_one_hot = F.one_hot(labels.view(-1), self.num_classes).float() * self.m
        value = self.s * (xw_norm - label_one_hot)
        return F.cross_entropy(input=value, target=labels.view(-1))