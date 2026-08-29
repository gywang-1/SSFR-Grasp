# engine/engine.py
import os
import time
import json  # NEW
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize, overlay_davis
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)


def _is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def _get_rank():
    return dist.get_rank() if _is_dist_avail_and_initialized() else 0


def _get_world_size():
    return dist.get_world_size() if _is_dist_avail_and_initialized() else 1

def _all_reduce_tensor(t: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    对单个 tensor 做分布式 all_reduce；若未初始化分布式，直接原样返回。
    """
    if _is_dist_avail_and_initialized():
        t = t.clone()
        dist.all_reduce(t, op=op)
    return t

def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    thresh = 0.5
    model.train()
    time.sleep(2)
    end = time.time()

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).unsqueeze(1)

        with amp.autocast():
            pred, target, loss = model(image, text, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()
        # ===== EMA update (after optimizer step) =====
        with torch.no_grad():
            bare = _unwrap_ddp(model)
            ema_helper = getattr(bare, "ema_helper", None)
            if ema_helper is not None and bool(getattr(args, "USE_EMA_TEACHER", True)):
                ema_helper.update(bare)

        # metrics (DDP all-reduce)
        iou, pr5 = trainMetricGPU(pred, target, thresh, 0.5)

        if _is_dist_avail_and_initialized():
            dist.all_reduce(loss.detach()); dist.all_reduce(iou); dist.all_reduce(pr5)
            world = _get_world_size()
            loss = loss / world; iou = iou / world; pr5 = pr5 / world

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end); end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if _get_rank() in [-1, 0, 0]:
                wandb.log(
                    {
                        "time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/loss_total": loss_meter.val,
                        "training/iou": iou_meter.val,
                        "training/prec@50": pr_meter.val,
                    },
                    step=epoch * len(train_loader) + (i + 1))


# === helpers ===
def _unwrap_ddp(m):
    return m.module if hasattr(m, "module") else m


@torch.no_grad()
def _get_model_for_eval(model: torch.nn.Module, args=None, epoch=None) -> torch.nn.Module:
    """
    评测用模型选择：
    - 若 args.USE_EMA_TEACHER=True 且存在 EMA 教师且 epoch >= EMA_EVAL_FROM_EPOCH -> 用 EMA
    - 否则回退到学生模型
    """
    bare = _unwrap_ddp(model)
    ema_teacher = getattr(getattr(bare, "ema_helper", None), "teacher", None)

    use_ema = bool(getattr(args, "USE_EMA_TEACHER", True))
    # 默认第 1 个 epoch 起使用 EMA（ViT 更稳）
    warm_from = int(getattr(args, "EMA_EVAL_FROM_EPOCH", 8))  # CHANGED: default 1

    use_ema_now = (use_ema and (ema_teacher is not None) and
                   (epoch is None or epoch >= warm_from))

    eval_model = ema_teacher if use_ema_now else bare
    eval_model.eval()
    return eval_model


@torch.no_grad()
def _ensure_same_device(module: torch.nn.Module, device: torch.device) -> None:
    """
    兜底把 module 以及其 state_dict 中残留在其他设备上的权重/缓冲，全部迁移到 device。
    防止 “Expected all tensors to be on the same device ...” 一类问题。
    """
    module.to(device)
    sd = module.state_dict()
    moved = False
    for k, v in sd.items():
        if v.device != device:
            sd[k] = v.to(device)
            moved = True
    if moved:
        module.load_state_dict(sd, strict=False)


@torch.no_grad()
def validate(val_loader, model, epoch, args):
    """
    validate() 返回 (iou, oIoU, prec)
    - 仅保留单尺度推理
    - auto_calibrate_thresh=True 时，按“全验证集 / 多卡聚合后”的 oIoU 选最佳阈值
    """
    iou_list = []

    key = f"{args.dataset}_{args.exp_name}"
    th_file = os.path.join(args.output_dir, "best_thresholds.json")
    auto_cal = bool(getattr(args, "auto_calibrate_thresh", True))
    default_th = float(getattr(args, "threshold", 0.5))

    # 候选阈值
    cand = torch.arange(0.35, 0.651, 0.02).tolist() if auto_cal else [default_th]
    num_cand = len(cand)

    # 用 tensor 而不是 Python dict，便于多卡 all_reduce
    I_buckets = torch.zeros(num_cand, dtype=torch.float64)
    U_buckets = torch.zeros(num_cand, dtype=torch.float64)

    eval_model = _get_model_for_eval(model, args=args, epoch=epoch)

    time.sleep(2)

    I = []
    U = []
    first_batch = True
    device_for_reduce = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for imgs, texts, param in val_loader:
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)

        if first_batch:
            _ensure_same_device(eval_model, imgs.device)
            device_for_reduce = imgs.device
            I_buckets = I_buckets.to(device_for_reduce)
            U_buckets = U_buckets.to(device_for_reduce)
            first_batch = False

        preds, _ = eval_model(imgs, texts)
        preds = torch.sigmoid(preds)

        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds, size=imgs.shape[-2:], mode='bicubic', align_corners=False)
        preds = preds.squeeze(1)  # [B, H, W]

        for pred, mask_dir, mat, ori_size in zip(preds, param['mask_dir'], param['inverse'], param['ori_size']):
            h, w = np.array(ori_size)
            mat = np.array(mat)

            pred = pred.detach().float().cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=0.)

            mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"GT mask not found: {mask_dir}, skipping sample.")
                continue

            gt_mask = (mask.astype(np.float32) / 255.0).astype(np.uint8)

            # 默认阈值：用于当前 validate 日志中的 IoU / oIoU / Prec
            pred_bin = (pred > default_th).astype(np.uint8)
            inter = np.logical_and(pred_bin, gt_mask)
            union = np.logical_or(pred_bin, gt_mask)

            I.append(np.sum(inter))
            U.append(np.sum(union))
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

            # 自动阈值搜索：累计每个阈值在全验证集上的 I/U
            if auto_cal:
                for idx, th in enumerate(cand):
                    pred_tmp = (pred > th).astype(np.uint8)
                    inter_t = np.logical_and(pred_tmp, gt_mask)
                    union_t = np.logical_or(pred_tmp, gt_mask)
                    I_buckets[idx] += float(inter_t.sum())
                    U_buckets[idx] += float(union_t.sum())

    if len(iou_list) == 0:
        logger.warning("validate(): no valid samples were processed; returning zeros for metrics.")
        iou_val = 0.0
        oIoU_val = 0.0
        prec = {f'Pr@{t * 10}': 0.0 for t in range(5, 10)}
        logger.info(
            'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  oIoU:{:.2f}'.format(
                epoch, args.epochs, 100. * iou_val, 100. * oIoU_val
            )
        )
        return iou_val, oIoU_val, prec

    # ===== 主指标：全卡聚合 =====
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(device_for_reduce)
    iou_list = concat_all_gather(iou_list)

    I = torch.from_numpy(np.stack(I)).to(device_for_reduce)
    I = concat_all_gather(I).sum()

    U = torch.from_numpy(np.stack(U)).to(device_for_reduce)
    U = concat_all_gather(U).sum()

    oIoU = (I / (U + 1e-12)).item()

    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)

    iou = iou_list.mean().item()
    prec = {f'Pr@{(i + 5) * 10}': prec_list[i].item() for i in range(len(prec_list))}

    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  oIoU:{:.2f}'.format(
        epoch, args.epochs, 100. * iou, 100. * oIoU
    )
    temp = '  ' + '  '.join([f"{k}: {100. * v:.2f}" for k, v in prec.items()])
    logger.info(head + temp)

    # ===== 自动阈值：对各候选阈值的 I/U 做全局 all_reduce 后再选 =====
    if auto_cal:
        I_buckets = _all_reduce_tensor(I_buckets)
        U_buckets = _all_reduce_tensor(U_buckets)

        if _get_rank() == 0:
            best_th = default_th
            best_oiou = -1.0

            I_np = I_buckets.detach().cpu().numpy()
            U_np = U_buckets.detach().cpu().numpy()

            for idx, th in enumerate(cand):
                if U_np[idx] <= 0:
                    continue
                oiou = float(I_np[idx] / (U_np[idx] + 1e-12))
                if oiou > best_oiou:
                    best_oiou = oiou
                    best_th = float(th)

            try:
                db = {}
                if os.path.isfile(th_file):
                    with open(th_file, "r") as f:
                        db = json.load(f)
                db[key] = best_th
                with open(th_file, "w") as f:
                    json.dump(db, f, indent=2)
                logger.info(f"[THRESH] calibrated(global) {key} => {best_th:.2f} (by oIoU={best_oiou:.6f})")
            except Exception as e:
                logger.warning(f"[THRESH] save failed: {e}")
    else:
        if _get_rank() == 0:
            logger.info(f"[THRESH] auto calibration disabled, validate uses fixed threshold: {default_th:.2f}")

    return iou, oIoU, prec


# === inference：完整替换 ===
@torch.no_grad()
def inference(test_loader, model, args):
    """
    inference() 返回 (iou, oIoU, prec)
    - 评测优先使用 EMA teacher（若存在）
    - 逐句预测并计算 IoU
    - 读取并使用已校准阈值（若存在）
    """
    iou_list = []

    # 读取校准阈值（若存在）
    # 读取校准阈值（仅在 auto_calibrate_thresh=True 时启用）
    key = f"{args.dataset}_{args.exp_name}"
    th_file = os.path.join(args.output_dir, "best_thresholds.json")
    thresh = float(getattr(args, "threshold", 0.5))
    auto_cal = bool(getattr(args, "auto_calibrate_thresh", True))

    if auto_cal and os.path.isfile(th_file):
        try:
            with open(th_file, "r") as f:
                db = json.load(f)
            if key in db:
                thresh = float(db[key])
                logger.info(f"[THRESH] use calibrated threshold for {key}: {thresh:.2f}")
            else:
                logger.info(
                    f"[THRESH] no calibrated threshold found for {key}, fallback to fixed threshold: {thresh:.2f}")
        except Exception as e:
            logger.warning(f"[THRESH] load failed, fallback to {thresh:.2f}: {e}")
    else:
        logger.info(f"[THRESH] auto calibration disabled or threshold file missing, use fixed threshold: {thresh:.2f}")

    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    I = []
    U = []

    # 推理优先使用 EMA 教师
    eval_model = _get_model_for_eval(model, args=args, epoch=None)
    eval_model.eval()
    time.sleep(2)
    first_batch = True
    device_for_reduce = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for img, param in tbar:
        img = img.cuda(non_blocking=True)
        if first_batch:
            _ensure_same_device(eval_model, img.device)
            device_for_reduce = img.device
            first_batch = False

        # 读取并归一化一次 GT mask
        mask_path = param['mask_dir'][0]
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"GT mask not found: {mask_path}; skipping sample.")
            continue
        gt_mask_base = (mask.astype(np.float32) / 255.0).astype(np.uint8)

        if args.visualize:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id); mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name), img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name), img=mask)
            mask_vis_name = '{}-mask_vis.png'.format(seg_id)
            new_mask = gt_mask_base.astype(np.uint8)
            mask_vis = overlay_davis(param['ori_img'][0].cpu().numpy(), new_mask)
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_vis_name), img=mask_vis)

        for sent in param['sents']:
            # prepare text tensor
            text = tokenize(sent, args.word_len, True).cuda(non_blocking=True)

            # model forward
            pred, attn = eval_model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred, size=img.shape[-2:], mode='bicubic', align_corners=False)
            pred = pred.squeeze()  # H,W

            h, w = param['ori_size'].numpy()[0]; mat = param['inverse'].numpy()[0]
            pred_np = pred.detach().cpu().numpy()
            pred_np = cv2.warpAffine(pred_np, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=0.)
            pred_bin = (pred_np > thresh).astype(np.uint8)

            gt_mask = gt_mask_base  # reuse

            inter = np.logical_and(pred_bin, gt_mask)
            union = np.logical_or(pred_bin, gt_mask)
            I.append(np.sum(inter)); U.append(np.sum(union))
            iou = np.sum(inter) / (np.sum(union) + 1e-6); iou_list.append(iou)

            if args.visualize:
                ori_img = param['ori_img'][0].cpu().numpy()
                try:
                    attn_map = cv2.warpAffine(attn, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=0.)
                    maps_vis = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    maps_vis = (0.5 * cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR) + 0.5 * maps_vis).astype(np.uint8)
                    attn_name = '{}-iou={:.2f}-{}-attn.png'.format(seg_id, iou * 100, sent)
                    cv2.imwrite(filename=os.path.join(args.vis_dir, attn_name), img=maps_vis)
                except Exception:
                    pass
                pred_u8 = np.array(pred_bin * 255, dtype=np.uint8)
                sent_name = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou * 100, sent_name)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name), img=pred_u8)
                mask_vis_name = '{}-iou={:.2f}-{}_maskvis.png'.format(seg_id, iou * 100, sent_name)
                new_pred = (pred_u8 / 255).astype(np.uint8)
                mask_vis = overlay_davis(param['ori_img'][0].cpu().numpy(), new_pred)
                cv2.imwrite(filename=os.path.join(args.vis_dir, mask_vis_name), img=mask_vis)

    # Metric aggregation & safe-return
    logger.info('=> Metric Calculation <=')
    if len(iou_list) == 0:
        logger.warning("inference(): no valid predictions were produced; returning zeros for metrics.")
        iou_val = 0.0
        oIoU_val = 0.0
        prec = {f'Pr@{t*10}': 0.0 for t in range(5, 10)}
        logger.info('IoU={:.2f}'.format(100. * iou_val))
        logger.info('oIoU={:.2f}'.format(100. * oIoU_val))
        for k, v in prec.items():
            logger.info('{}: {:.2f}.'.format(k, 100. * v))
        return iou_val, oIoU_val, prec

    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(device_for_reduce)
    I = torch.from_numpy(np.stack(I)).to(device_for_reduce); I = I.sum()
    U = torch.from_numpy(np.stack(U)).to(device_for_reduce); U = U.sum()
    iou_list = concat_all_gather(iou_list)

    oIoU = (I / (U + 1e-12)).item()
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean(); prec_list.append(tmp)
    iou = iou_list.mean().item()
    prec = {f'Pr@{(i+5)*10}': prec_list[i].item() for i in range(len(prec_list))}

    logger.info('IoU={:.2f}'.format(100. * iou))
    logger.info('oIoU={:.2f}'.format(100. * oIoU))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100. * v))
    return iou, oIoU, prec


@torch.no_grad()
def inference_vit(test_loader, model, args):
    """
    ViT 推理路径：复用 inference() 的逻辑以保证一致性。
    """
    return inference(test_loader, model, args)
