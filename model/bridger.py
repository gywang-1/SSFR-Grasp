'''
Code manily adapted from https://github.com/kkakkkka/ETRIS/blob/main/model/bridger.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import VLSAadapter
import math

def interpolate_pos_embed(pos_embed: torch.Tensor, hw: tuple[int, int], has_cls: bool = True):
    """
    兼容 2D([L,C]) / 3D([1,L,C]) 的 ViT 位置编码，并在长度匹配时快速返回。
    hw: (H, W) 为当前 patch 网格大小
    返回形状与输入维度一致（输入 2D -> 2D，输入 3D -> 3D）
    """
    # 统一到 3D 处理
    squeeze_back = False
    if pos_embed.ndim == 2:           # [L, C] -> [1, L, C]
        pos_embed = pos_embed.unsqueeze(0)
        squeeze_back = True
    elif pos_embed.ndim != 3:
        raise ValueError(f"pos_embed ndim must be 2 or 3, got {pos_embed.ndim}")

    Bpe, Lpe, C = pos_embed.shape     # Bpe 应该为 1
    H, W = hw
    want_L = (H * W + 1) if has_cls else (H * W)

    # 若长度本来就匹配，直接返回（避免数值微扰 & 提速）
    if Lpe == want_L:
        return pos_embed.squeeze(0) if squeeze_back else pos_embed

    # 拆 cls / grid
    if has_cls:
        cls_pos = pos_embed[:, :1, :]            # (1,1,C)
        grid_pos = pos_embed[:, 1:, :]           # (1,N,C)
    else:
        cls_pos, grid_pos = None, pos_embed      # (1,N,C)

    n = grid_pos.shape[1]
    gs_old = int(round(math.sqrt(n)))
    if gs_old * gs_old != n:
        raise ValueError(f"grid tokens must be square, got N={n}")

    # (1,N,C) -> (1,C,gs,gs)
    grid_pos = grid_pos.reshape(1, gs_old, gs_old, C).permute(0, 3, 1, 2)
    # 插值到 (H, W)
    grid_pos = F.interpolate(grid_pos, size=(H, W), mode="bicubic", align_corners=False)
    # (1,C,H,W) -> (1,H*W,C)
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, H * W, C)

    out = torch.cat([cls_pos, grid_pos], dim=1) if has_cls else grid_pos
    return out.squeeze(0) if squeeze_back else out

# add forward output, current best, no change; prenorm
class Bridger_SA_RN_fwd(nn.Module):
    def __init__(self,
                 d_img = [512, 1024, 2048],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [2, 1, 2],
                 num_layers = 12,
                 fusion_stage = 3,
                 word_dim = 1024
                ):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.fusion_stage = fusion_stage

        self.fusion= nn.ModuleList()
        self.zoom_in = nn.ModuleList()
        self.zoom_out = nn.ModuleList()
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.ln_v = nn.ModuleList()
        self.ln_t = nn.ModuleList()
        for i in range(num_stages):
            if i >= num_stages - fusion_stage:
                self.fusion.append(VLSAadapter(d_model=d_model, nhead=nhead))
                if i < num_stages - 1:
                    self.zoom_in.append(nn.Conv2d(d_img[i], d_model, kernel_size=strides[i], stride=strides[i], bias=False))
                    self.zoom_out.append(nn.ConvTranspose2d(d_model, d_img[i], kernel_size=strides[i], stride=strides[i], bias=False))
                    self.linear1.append(nn.Linear(d_txt, d_model))
                    self.linear2.append(nn.Linear(d_model, d_txt))
                    self.ln_v.append(nn.LayerNorm(d_model))
                    self.ln_t.append(nn.LayerNorm(d_model))
                else:
                    self.zoom_in.append(nn.ConvTranspose2d(d_img[i], d_model ,kernel_size=strides[i], stride=strides[i], bias=False))
                    self.zoom_out.append(nn.Conv2d(d_model, d_img[i], kernel_size=strides[i], stride=strides[i], bias=False))
                    self.linear1.append(nn.Linear(d_txt, d_model))
                    self.linear2.append(nn.Linear(d_model, d_txt))
                    self.ln_v.append(nn.LayerNorm(d_model))
                    self.ln_t.append(nn.LayerNorm(d_model))
            else:
                self.fusion.append(None)
                self.zoom_in.append(None)
                self.zoom_out.append(None)
                self.linear1.append(None)
                self.linear2.append(None)
                self.ln_v.append(None)
                self.ln_t.append(None)
        # change last_conv for res50 : outchannels 1024, res101: outchannels 512
        self.last_conv = nn.Conv2d(d_model, word_dim, kernel_size=strides[-1], stride=strides[-1], bias=False)
        self.last_linear = nn.Linear(d_model, d_txt)
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, vis, text, backbone, pad_mask):
        # ---- 保证 dtype/device 一致性（关键） ----
        vis_enc = backbone.visual
        target_dtype = vis_enc.conv1.weight.dtype
        target_device = vis_enc.conv1.weight.device

        vis = vis.to(device=target_device, dtype=target_dtype)

        def stem(x):
            for conv, bn in [(vis_enc.conv1, vis_enc.bn1), (vis_enc.conv2, vis_enc.bn2),
                             (vis_enc.conv3, vis_enc.bn3)]:
                x = vis_enc.relu(bn(conv(x)))
            x = vis_enc.avgpool(x)
            return x

        # vision
        vis = stem(vis)
        vis = vis_enc.layer1(vis)
        vis_enc_layers = [vis_enc.layer2, vis_enc.layer3, vis_enc.layer4]

        # language
        # ---- 关键：先把 text/pad_mask 放到同一 device，再 token_embedding ----
        text = text.to(device=target_device, non_blocking=True)
        if pad_mask is not None:
            pad_mask = pad_mask.to(device=target_device, non_blocking=True)

        txt = backbone.token_embedding(text).to(dtype=backbone.dtype, device=target_device)  # [B, L, C]
        txt_enc = backbone.transformer
        pos = backbone.positional_embedding.to(dtype=backbone.dtype, device=target_device)[:txt.size(1)]
        txt = txt + pos
        txt = txt.permute(1, 0, 2).contiguous()  # [B,L,C] -> [L,B,C] 供 resblocks 使用

        # fusion
        stage_i = 0
        vis_outs = []
        forward_out = []
        last_v, last_t = None, None
        for i in range(self.num_layers):
            if (i + 1) % 4 != 0:
                txt = txt_enc.resblocks[i](txt)
            else:
                txt = txt_enc.resblocks[i](txt)
                vis = vis_enc_layers[stage_i](vis)

                if stage_i >= self.num_stages - self.fusion_stage:
                    v = vis.clone()
                    t = txt.clone()  # t: L, B, dim

                    v = self.zoom_in[stage_i](v)
                    t = self.linear1[stage_i](t)

                    B, C, H, W = v.shape
                    v = v.reshape(B, C, -1).permute(2, 0, 1)  # -> HW, B, C

                    if stage_i == 0 or last_v is None or last_t is None:
                        v, t = self.ln_v[stage_i](v), self.ln_t[stage_i](t)
                    else:
                        v, t = self.ln_v[stage_i](v + last_v), self.ln_t[stage_i](t + last_t)

                    v, t = self.fusion[stage_i](v, t)
                    last_v, last_t = v, t

                    v = v.permute(1, 2, 0).reshape(B, -1, H, W)  # -> B, C, H, W

                    if stage_i == 2:
                        forward_out.append(v)
                        forward_out.append(t)

                    v = self.zoom_out[stage_i](v)
                    t = self.linear2[stage_i](t)

                    vis = vis + v
                    txt = txt + t

                stage_i += 1
                if stage_i < self.num_stages:
                    vis_outs.append(vis)

        # After fusion
        vis = vis_enc.attnpool(vis)

        # forward_out[0]: B,C',H,W ; forward_out[1]: L,B,C'
        forward_vis = self.last_conv(forward_out[0]).to(vis.device, vis.dtype)
        vis = vis + forward_vis
        forward_t = self.last_linear(forward_out[1]).to(txt.device, txt.dtype)
        txt = txt + forward_t

        vis_outs.append(vis)

        # language
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        txt = backbone.ln_final(txt).type(backbone.dtype)


        # eot embedding
        state = txt[torch.arange(txt.shape[0]), text.argmax(dim=-1)] @ backbone.text_projection

        return vis_outs, txt, state


class Bridger_SA_ViT(nn.Module):
    def __init__(self,
                 d_img=[768, 768, 768],
                 d_txt=512,
                 d_model=64,
                 nhead=8,
                 num_stages=3,
                 strides=[2, 2, 2],
                 num_layers=12,
                 shared_weights=False,
                 ):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers

        self.zoom_in, self.zoom_out = nn.ModuleList(), nn.ModuleList()
        self.linear1, self.linear2 = nn.ModuleList(), nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.ln_v = nn.ModuleList()
        self.ln_t = nn.ModuleList()
        for i in range(num_stages):
            self.fusion.append(VLSAadapter(d_model=d_model, nhead=nhead))
            self.linear1.append(nn.Linear(d_txt, d_model))
            self.linear2.append(nn.Linear(d_model, d_txt))
            self.zoom_in.append(nn.Linear(d_img[i], d_model))
            self.zoom_out.append(nn.Linear(d_model, d_img[i]))
            self.ln_v.append(nn.LayerNorm(d_model))
            self.ln_t.append(nn.LayerNorm(d_model))
        self.last_conv = nn.Linear(d_model, 512)
        self.last_linear = nn.Linear(d_model, d_txt)
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, text, backbone, pad_mask=None):
        # ---- 保证 dtype/device 一致性（关键） ----
        vis_enc = backbone.visual
        img = img.to(device=vis_enc.conv1.weight.device, dtype=backbone.dtype)

        # vision
        x = vis_enc.conv1(img)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [B, width, grid^2]
        x = x.permute(0, 2, 1)                      # [B, grid^2, width]
        """
        cls = vis_enc.class_embedding.to(dtype=x.dtype, device=x.device)
        x = torch.cat([cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + vis_enc.positional_embedding.to(dtype=x.dtype, device=x.device)
        """""
        # 现有代码
        cls = vis_enc.class_embedding.to(dtype=x.dtype, device=x.device)
        x = torch.cat([cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        # 计算当前 token 网格 (去掉 cls)
        N = x.shape[1] - 1
        H = int(round(N ** 0.5))
        W = N // H
        assert H * W == N, f"token count must form a square grid, got N={N}"

        # 取出原始位置编码（兼容 2D/3D）
        pos_raw = vis_enc.positional_embedding.to(dtype=x.dtype, device=x.device)
        # 计算“期望长度”
        want_L = (H * W + 1)

        # 若长度一致，直接相加；否则插值到 (H,W) 再相加
        if (pos_raw.shape[-2] if pos_raw.ndim == 3 else pos_raw.shape[-2]) == want_L:
            pos = pos_raw if pos_raw.ndim == 3 else pos_raw.unsqueeze(0)  # broadcast 到 batch 维
        else:
            pos = interpolate_pos_embed(pos_raw, (H, W), has_cls=True)

        x = x + (pos if pos.ndim == 3 else pos.unsqueeze(0))

        x = vis_enc.ln_pre(x)
        vis = x.permute(1, 0, 2)  # NLD -> LND

        # language
        txt = backbone.token_embedding(text).to(dtype=backbone.dtype, device=vis.device)
        txt_enc = backbone.transformer
        pos = backbone.positional_embedding.to(dtype=backbone.dtype, device=vis.device)[:txt.size(1)]
        txt = txt + pos
        txt = txt.permute(1, 0, 2)  # NLD -> LND

        # fusion
        stage_i = 0
        vis_outs, forward_out = [], []
        last_v, last_t = None, None
        for i in range(self.num_layers):
            if (i + 1) % 4 != 0:
                vis = vis_enc.transformer.resblocks[i](vis)
                txt = txt_enc.resblocks[i](txt)
            else:
                vis = vis_enc.transformer.resblocks[i](vis)
                txt = txt_enc.resblocks[i](txt)

                v = vis.clone()   # LND
                t = txt.clone()

                v = self.zoom_in[stage_i](v)
                t = self.linear1[stage_i](t)

                if stage_i == 0 or last_v is None or last_t is None:
                    v, t = self.ln_v[stage_i](v), self.ln_t[stage_i](t)
                else:
                    v, t = self.ln_v[stage_i](v + last_v), self.ln_t[stage_i](t + last_t)

                v, t = self.fusion[stage_i](v, t)
                last_v, last_t = v, t

                if stage_i == 2:
                    forward_out.append(v)
                    forward_out.append(t)

                v = self.zoom_out[stage_i](v)
                t = self.linear2[stage_i](t)

                vis = vis + v
                txt = txt + t

                stage_i += 1
                if stage_i < self.num_stages:
                    vis_out = vis[1:, :, :].permute(1, 2, 0)  # N, D, L
                    B, C, L = vis_out.shape
                    H = int(L ** 0.5)
                    W = L // H
                    vis_out = vis_out.reshape(B, C, H, W)     # B, D, H, W
                    vis_out = F.interpolate(vis_out, scale_factor=2 // stage_i, mode='bilinear')
                    vis_outs.append(vis_out)

        # After fusion
        vis = vis.permute(1, 0, 2)                 # LND -> NLD
        vis = vis_enc.ln_post(vis)

        if vis_enc.proj is not None:
            vis = vis @ vis_enc.proj

        B, N, C = vis[:, 1:, :].shape
        H = int(N ** 0.5)
        W = N // H
        vis = vis[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)

        # forward_out
        forward_vis = self.last_conv(forward_out[0]).to(vis.device, vis.dtype)
        forward_vis = forward_vis[1:, ...].reshape(B, C, H, W)
        vis = vis + forward_vis
        forward_t = self.last_linear(forward_out[1]).to(txt.device, txt.dtype)
        txt = txt + forward_t

        vis_outs.append(vis)

        # language
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        txt = backbone.ln_final(txt).type(backbone.dtype)

        state = txt[torch.arange(txt.shape[0]), text.argmax(dim=-1)] @ backbone.text_projection

        return vis_outs, txt, state
