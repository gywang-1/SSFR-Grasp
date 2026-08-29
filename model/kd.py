# model/kd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

#VIT-B
class KDLoss(nn.Module):
    def __init__(self, T=3.0, w_logit=0.4, w_feat=0.2, w_attn=0.05,
                 feat_norm='l2', ramp_steps=2000, warmup_steps=300):
        super().__init__()
        self.T = float(T)
        self.w_logit = float(w_logit)
        self.w_feat = float(w_feat)
        self.w_attn = float(w_attn)
        self.feat_norm = feat_norm
        self.ramp_steps = ramp_steps
        self.warmup_steps = warmup_steps
        self.step = 0

    def update_step(self):
        self.step += 1

    def ramp_weight(self, base_weight):
        if self.step < self.warmup_steps:
            return 0.0
        elif self.step >= self.ramp_steps:
            return base_weight
        else:
            p = (self.step - self.warmup_steps) / (self.ramp_steps - self.warmup_steps)
            return base_weight * (0.5 - 0.5 * math.cos(p * math.pi))

    def kl_logits(self, zs, zt):
        if self.w_logit == 0.0:
            return zs.new_zeros([])
        T = self.T
        ps = torch.sigmoid(zs / T).clamp(1e-6, 1 - 1e-6)
        pt = torch.sigmoid(zt / T).clamp(1e-6, 1 - 1e-6)
        kl = pt * (pt / ps).log() + (1 - pt) * ((1 - pt) / (1 - ps)).log()
        return kl.mean() * (T * T)

    def feat_loss(self, fs, ft):
        if self.w_feat == 0.0:
            return fs.new_zeros([])
        fs_n = F.normalize(fs, dim=1)
        ft_n = F.normalize(ft, dim=1)
        return F.mse_loss(fs_n, ft_n)

    def attn_loss(self, as_, at):
        if self.w_attn == 0.0:
            return as_.new_zeros([])
        return F.l1_loss(as_, at)

    def forward(self, zs, zt, fs, ft, as_=None, at=None):
        loss = self.w_logit * self.kl_logits(zs, zt)
        w_feat_now = self.ramp_weight(self.w_feat)
        w_attn_now = self.ramp_weight(self.w_attn)
        loss += w_feat_now * self.feat_loss(fs, ft)
        if as_ is not None and at is not None:
            loss += w_attn_now * self.attn_loss(as_, at)
        self.update_step()
        return loss

#Resnet
# class KDLoss(nn.Module):
#     """
#     更稳健的蒸馏损失：
#     - logits 蒸馏：使用基于 sigmoid 概率的“二元 KL”(per-pixel)
#     - 特征蒸馏：L2 / 余弦（默认关闭，后续可渐进开启）
#     - 注意力蒸馏：L1（默认关闭）
#     """
#     def __init__(self, T=2.0, w_logit=0.1, w_feat=0.0, w_attn=0.0, feat_norm='l2'):
#         super().__init__()
#         self.T = float(T)
#         self.w_logit = float(w_logit)
#         self.w_feat = float(w_feat)
#         self.w_attn = float(w_attn)
#         self.feat_norm = feat_norm
#
#     def kl_logits(self, zs, zt):
#         """
#         zs, zt: B×1×H×W 的 logits
#         使用 per-pixel 的二元 KL：
#           KL(q||p) = q*log(q/p) + (1-q)*log((1-q)/(1-p))
#         这里 q = sigmoid(zt/T), p = sigmoid(zs/T)
#         """
#         if self.w_logit == 0.0:
#             return zs.new_zeros([])
#         T = self.T
#         ps = torch.sigmoid(zs / T)
#         pt = torch.sigmoid(zt / T)
#         eps = 1e-6
#         ps = ps.clamp(eps, 1 - eps)
#         pt = pt.clamp(eps, 1 - eps)
#         kl = pt * (pt / ps).log() + (1 - pt) * ((1 - pt) / (1 - ps)).log()
#         # 与传统 KD 一致，乘以 T^2 做温度缩放
#         return kl.mean() * (T * T)
#
#     def feat_loss(self, fs, ft):
#         # fs, ft: B×C×H×W
#         if self.w_feat == 0.0:
#             return fs.new_zeros([])
#         if self.feat_norm == 'cosine':
#             fs_n = F.normalize(fs.flatten(2), dim=1)
#             ft_n = F.normalize(ft.flatten(2), dim=1)
#             return 1 - (fs_n * ft_n).sum(dim=1).mean()
#         # 默认 L2
#         return F.mse_loss(fs, ft)
#
#     def attn_loss(self, as_, at):
#         # as_, at: B×1×H×W
#         if self.w_attn == 0.0:
#             return as_.new_zeros([])
#         return F.l1_loss(as_, at)
#
#     def forward(self, zs, zt, fs, ft, as_=None, at=None):
#         loss = self.w_logit * self.kl_logits(zs, zt)
#         loss += self.w_feat * self.feat_loss(fs, ft)
#         if as_ is not None and at is not None:
#             loss += self.w_attn * self.attn_loss(as_, at)
#         return loss


class EMATeacher:
    """
    EMA 教师：深拷贝 student，并在同一 device 上维护滑动平均权重。
    - 只对浮点张量做 EMA（跳过 long/bool，如 num_batches_tracked）
    - 彻底避免 CPU/GPU 混用与 dtype 冲突
    """
    def __init__(self, student: nn.Module, momentum: float = 0.9995):
        self.m = float(momentum)
        device = next(student.parameters()).device

        # 深拷贝并移动到与 student 相同的 device
        self.teacher = deepcopy(student).eval()
        self.teacher.to(device)

        # 确保 state_dict 全部在 device
        tsd = self.teacher.state_dict()
        tsd = {k: v.to(device) for k, v in tsd.items()}
        self.teacher.load_state_dict(tsd)

        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student: nn.Module):
        msd = student.state_dict()
        tsd = self.teacher.state_dict()
        device = next(student.parameters()).device

        # 再保险：把 teacher 的所有张量搬到正确的 device
        for k in list(tsd.keys()):
            if tsd[k].device != device:
                tsd[k] = tsd[k].to(device)

        # 仅对浮点张量进行 EMA（跳过 long/bool 计数器等）
        for k in tsd.keys():
            if k not in msd:
                continue
            tv = tsd[k]
            mv = msd[k]
            if mv.shape != tv.shape:
                continue
            if not torch.is_floating_point(tv):
                continue
            if mv.device != device:
                mv = mv.to(device)
            tv.data.mul_(self.m).add_(mv.data, alpha=1.0 - self.m)

        self.teacher.load_state_dict(tsd, strict=True)
