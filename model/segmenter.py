# model/segmenter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from model.clip import build_model
from .layers import Projector, FPN, FPN_vit
from .bridger import Bridger_SA_ViT, Bridger_SA_RN_fwd
from .decoder import TransformerDecoder
from .kd import KDLoss, EMATeacher
from .afe import AFEBlock


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1)) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return bce + dice


def _make_decoder(cfg):
    decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                 d_model=cfg.vis_dim,
                                 nhead=cfg.num_head,
                                 dim_ffn=cfg.dim_ffn,
                                 dropout=cfg.dropout,
                                 return_intermediate=cfg.intermediate)
    return decoder


def _make_fpn_and_proj(cfg, vit=False):
    if vit:
        fpn = FPN_vit(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, language_fuser=True, decoding=False)
    else:
        fpn = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, language_fuser=True, decoding=False)
    proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)  # cfg.vis_dim=512
    return fpn, proj


class _KDBase(nn.Module):

    def train(self, mode: bool = True):
        # 1) 学生照常切换
        super().train(mode)

        # 2) teacher / ema teacher 永远 eval（不参与 BN/Dropout 的训练态）
        teacher = getattr(self, "teacher", None)
        if teacher is not None:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False

        ema_helper = getattr(self, "ema_helper", None)
        ema_teacher = getattr(ema_helper, "teacher", None) if ema_helper is not None else None
        if ema_teacher is not None:
            ema_teacher.eval()
            for p in ema_teacher.parameters():
                p.requires_grad = False

        return self

    def _setup_backbone_and_bridger(self, cfg):
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        if "RN" in cfg.clip_pretrain:
            backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
            bridger = Bridger_SA_RN_fwd(
                d_model=cfg.ladder_dim,
                nhead=cfg.nhead,
                fusion_stage=cfg.multi_stage,
                word_dim=getattr(cfg, 'word_dim', 1024)
            )
            vit = False
        else:
            backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size).float()
            bridger = Bridger_SA_ViT(d_model=cfg.ladder_dim, nhead=cfg.nhead)
            vit = True

        # Fix Backbone
        if "RN" in cfg.clip_pretrain:
            # RN：冻结全部编码器参数
           for _, p in backbone.named_parameters():
               p.requires_grad = False
        else:
          # ViT：仅解冻 positional_embedding
         for param_name, param in backbone.named_parameters():
                if 'positional_embedding' in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        return backbone, bridger, vit


    def _kd_prepare(self, cfg):
        # ------- KD/EMA 超参（更稳健的默认值） -------
        use_kd = getattr(cfg, 'USE_KD', True)
        use_ema = getattr(cfg, 'USE_EMA_TEACHER', True)
        T = float(getattr(cfg, 'KD_T', 2.0))
        w_logit = float(getattr(cfg, 'KD_W_LOGIT', 0.1))
        w_feat = float(getattr(cfg, 'KD_W_FEAT', 0.0))
        w_attn = float(getattr(cfg, 'KD_W_ATTN', 0.0))
        feat_norm = getattr(cfg, 'KD_FEAT_NORM', 'l2')
        ema_m = float(getattr(cfg, 'EMA_MOMENTUM', 0.999))
        teacher_ckpt = getattr(cfg, 'TEACHER_CHECKPOINT', "")

        # 蒸馏热身/渐进（防止前期性能暴跌）
        self.kd_warmup_steps = int(getattr(cfg, 'KD_WARMUP_STEPS', 1000))
        self.kd_ramp_steps   = int(getattr(cfg, 'KD_RAMP_STEPS',   4000))
        self.kd_weight_max   = float(getattr(cfg, 'KD_WEIGHT_MAX', 1.0))
        self.register_buffer('global_step', torch.zeros((), dtype=torch.long))

        self.use_kd = use_kd
        self.kd_loss = KDLoss(T=T, w_logit=w_logit, w_feat=w_feat, w_attn=w_attn, feat_norm=feat_norm)
        self.teacher = None
        self.ema_helper = None

        if (not self.use_kd) and use_ema:
            self.ema_helper = EMATeacher(self, momentum=ema_m)
            return

        if not self.use_kd:
            return

        # ------- 创建 teacher 时避免递归：禁用 KD/EMA 的 cfg 副本 -------
        teacher_cfg = copy.deepcopy(cfg)
        setattr(teacher_cfg, 'USE_KD', False)
        setattr(teacher_cfg, 'USE_EMA_TEACHER', False)

        self.teacher = type(self)(teacher_cfg)

        # 用学生权重初始化 teacher
        try:
            self.teacher.load_state_dict(self.state_dict(), strict=False)
        except Exception as e:
            print("Warning: teacher.load_state_dict from student failed:", e)

        # 暂不在此强行搬设备，交给 forward 前的 _ensure_teacher_device 二次校正
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # （可选）加载外部 teacher 权重
        if teacher_ckpt and len(teacher_ckpt) > 0:
            sd = torch.load(teacher_ckpt, map_location='cpu')
            t_sd = sd.get('state_dict', sd)
            try:
                self.teacher.load_state_dict(t_sd, strict=False)
            except Exception as e:
                print("Warning: loading external teacher_ckpt failed:", e)

        if use_ema:
            self.ema_helper = EMATeacher(self, momentum=ema_m)

    def _kd_update_ema(self):
        if self.ema_helper is not None:
            self.ema_helper.update(self)

    def _get_teacher(self):
        # 若启用 EMA，则优先使用 EMA teacher
        if self.ema_helper is not None:
            return self.ema_helper.teacher
        return self.teacher

    # 关键修复：在每次使用 teacher 前，强制把整套权重/缓冲迁移到与输入一致的 device
    def _ensure_teacher_device(self, device: torch.device):
        teacher = self._get_teacher()
        if teacher is None:
            return
        teacher.to(device)
        tsd = teacher.state_dict()
        moved = False
        for k, v in tsd.items():
            if v.device != device:
                tsd[k] = v.to(device)
                moved = True
        if moved:
            teacher.load_state_dict(tsd, strict=False)


class ETOG_res(_KDBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, self.bridger, self.is_vit = self._setup_backbone_and_bridger(cfg)
        self.input_dim = 1024
        self.batchnorm = cfg.batchnorm
        self.lang_fusion_type = cfg.lang_fusion_type
        self.bilinear = cfg.bilinear
        self.up_factor = 2 if self.bilinear else 1

        self.visual_sent_fpn, self.proj = _make_fpn_and_proj(cfg, vit=False)
        self.decoder = _make_decoder(cfg)
        # ===== AFEBlock (after FPN before decoder) =====
        self.use_afe = bool(getattr(cfg, "USE_AFE", False))
        if self.use_afe:
            n = int(getattr(cfg, "AFE_N", 2))  # 冲涨点：建议 2；更保守用 1
            dp = float(getattr(cfg, "AFE_DROP_PATH", 0.1))
            exp = int(getattr(cfg, "AFE_EXPAND", 4))
            ks = int(getattr(cfg, "AFE_K", 3))
            use_dmlp = bool(getattr(cfg, "AFE_DILATED_MLP", True))
            self.afe = nn.Sequential(*[
                AFEBlock(dim=cfg.vis_dim, drop_path=dp, expan_ratio=exp, kernel_size=ks, use_dilated_mlp=use_dmlp)
                for _ in range(n)
            ])
            gamma_init = float(getattr(cfg, "AFE_GAMMA_INIT", 0.5))
            self.afe_gamma = nn.Parameter(torch.ones(1) * gamma_init)

        self.loss = BCEDiceLoss()

        # KD / Teacher
        self._kd_prepare(cfg)

    def _forward_once(self, img, word):
        input_shape = img.shape[-2:]
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        im, word_ctx, state = self.bridger(img, word, self.backbone, pad_mask)  # vis feats & text tokens
        x_sent = self.visual_sent_fpn(im, state)  # B×C×H×W

        # ===== AFE inserted here =====
        if getattr(self, "use_afe", False):
            x_sent = x_sent + self.afe_gamma * self.afe(x_sent)

        B, _, H, W = x_sent.size()
        vis_feat, _ = self.decoder(x_sent, word_ctx, pad_mask=pad_mask)

        vis_feat = vis_feat.reshape(B, -1, H, W)                                # B×C×H×W
        mask_logits = self.proj(vis_feat, state)                                # B×1×H'×W'
        mask_logits = F.interpolate(mask_logits, input_shape, mode='bilinear', align_corners=False)
        return mask_logits, vis_feat  # 用于蒸馏的像素级多模态特征

    def _kd_weight_now(self):
        step = int(self.global_step.item())
        if step < self.kd_warmup_steps:
            return 0.0
        ramp = max(1, self.kd_ramp_steps)
        alpha = min(1.0, (step - self.kd_warmup_steps) / ramp)
        return self.kd_weight_max * alpha

    def forward(self, img, word, mask=None):
        if self.training:
            with torch.no_grad():
                self.global_step.add_(1)

            # 学生前向
            zs, fs = self._forward_once(img, word)
            sup_loss = self.loss(zs, mask)

            if not self.use_kd:
                return zs.detach(), mask, sup_loss

            # —— 关键：在 teacher 使用前强制对齐设备 —— #
            self._ensure_teacher_device(img.device)

            # 教师前向（EMA 优先）
            teacher_model = self._get_teacher()
            with torch.no_grad():
                zt, ft = teacher_model._forward_once(img, word)

            # 可选注意力图
            attn_s = fs.sum(dim=1, keepdim=True)
            attn_t = ft.sum(dim=1, keepdim=True)
            attn_s = F.interpolate(attn_s, mask.shape[-2:], mode='bilinear', align_corners=True)
            attn_t = F.interpolate(attn_t, mask.shape[-2:], mode='bilinear', align_corners=True)

            # 蒸馏损失（带热身与渐进权重）
            kd_core = self.kd_loss(zs, zt, fs, ft, attn_s, attn_t)
            kd = self._kd_weight_now() * kd_core

            loss_total = sup_loss + kd


            #self._kd_update_ema()

            return zs.detach(), mask, loss_total
        else:
            zs, fs = self._forward_once(img, word)
            interm = fs.sum(1, keepdim=True)
            attention_vis = F.interpolate(interm, img.shape[-2:], mode='bilinear', align_corners=True)
            attention_vis = torch.sigmoid(attention_vis)
            attention_vis = attention_vis.detach().cpu().numpy()[0, 0]
            return zs.detach(), attention_vis


class ETOG_vit(_KDBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, self.bridger, self.is_vit = self._setup_backbone_and_bridger(cfg)
        self.input_dim = 1024
        self.batchnorm = cfg.batchnorm
        self.lang_fusion_type = cfg.lang_fusion_type
        self.bilinear = cfg.bilinear
        self.up_factor = 2 if self.bilinear else 1

        self.visual_sent_fpn, self.proj = _make_fpn_and_proj(cfg, vit=True)
        self.decoder = _make_decoder(cfg)
        # ===== AFEBlock (after FPN before decoder) =====
        self.use_afe = bool(getattr(cfg, "USE_AFE", False))
        if self.use_afe:
            n = int(getattr(cfg, "AFE_N", 2))  # 冲涨点：建议 2；更保守用 1
            dp = float(getattr(cfg, "AFE_DROP_PATH", 0.1))
            exp = int(getattr(cfg, "AFE_EXPAND", 4))
            ks = int(getattr(cfg, "AFE_K", 3))
            use_dmlp = bool(getattr(cfg, "AFE_DILATED_MLP", True))
            self.afe = nn.Sequential(*[
                AFEBlock(dim=cfg.vis_dim, drop_path=dp, expan_ratio=exp, kernel_size=ks, use_dilated_mlp=use_dmlp)
                for _ in range(n)
            ])
            gamma_init = float(getattr(cfg, "AFE_GAMMA_INIT", 0.5))
            self.afe_gamma = nn.Parameter(torch.ones(1) * gamma_init)
        self.loss = BCEDiceLoss()

        self._kd_prepare(cfg)

    def _forward_once(self, img, word):
        input_shape = img.shape[-2:]
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        im, word_ctx, state = self.bridger(img, word, self.backbone, pad_mask)
        x_sent = self.visual_sent_fpn(im, state)
        # ===== AFE inserted here =====
        if getattr(self, "use_afe", False):
            x_sent = x_sent + self.afe_gamma * self.afe(x_sent)
        B, _, H, W = x_sent.size()
        vis_feat, _ = self.decoder(x_sent, word_ctx, pad_mask=pad_mask)
        vis_feat = vis_feat.reshape(B, -1, H, W)
        mask_logits = self.proj(vis_feat, state)
        mask_logits = F.interpolate(mask_logits, input_shape, mode='bilinear', align_corners=False)
        return mask_logits, vis_feat

    def _kd_weight_now(self):
        step = int(self.global_step.item())
        if step < self.kd_warmup_steps:
            return 0.0
        ramp = max(1, self.kd_ramp_steps)
        alpha = min(1.0, (step - self.kd_warmup_steps) / ramp)
        return self.kd_weight_max * alpha

    def forward(self, img, word, mask=None):
        if self.training:
            with torch.no_grad():
                self.global_step.add_(1)

            zs, fs = self._forward_once(img, word)
            sup_loss = self.loss(zs, mask)
            if not self.use_kd:
                return zs.detach(), mask, sup_loss

            # —— 关键：在 teacher 使用前强制对齐设备 —— #
            self._ensure_teacher_device(img.device)

            teacher_model = self._get_teacher()
            with torch.no_grad():
                zt, ft = teacher_model._forward_once(img, word)

            attn_s = fs.sum(dim=1, keepdim=True)
            attn_t = ft.sum(dim=1, keepdim=True)
            attn_s = F.interpolate(attn_s, mask.shape[-2:], mode='bilinear', align_corners=True)
            attn_t = F.interpolate(attn_t, mask.shape[-2:], mode='bilinear', align_corners=True)

            kd_core = self.kd_loss(zs, zt, fs, ft, attn_s, attn_t)
            kd = self._kd_weight_now() * kd_core

            loss_total = sup_loss + kd
            #self._kd_update_ema()
            return zs.detach(), mask, loss_total
        else:
            zs, fs = self._forward_once(img, word)
            return zs.detach(), None
