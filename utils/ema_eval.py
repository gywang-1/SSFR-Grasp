# utils/ema_eval.py
import torch

def get_model_for_eval(model: torch.nn.Module) -> torch.nn.Module:
    """
    返回用于验证/测试的模型实例：
    - 若存在 EMA 教师：返回 model.ema_helper.teacher
    - 否则：返回 model 自身
    并将其切换到 eval() 模式。
    """
    eval_model = getattr(getattr(model, "ema_helper", None), "teacher", None)
    if eval_model is None:
        eval_model = model
    eval_model.eval()
    return eval_model


@torch.no_grad()
def ensure_same_device(module: torch.nn.Module, device: torch.device) -> None:
    """
    兜底把 module 以及其 state_dict 中残留在其他设备上的权重/缓冲，全部迁移到 device。
    （与训练阶段的 _ensure_teacher_device 一致的目的）
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
def evaluate_refcoco(model: torch.nn.Module,
                     dataloader,
                     metric_fn,
                     device: torch.device,
                     *,
                     use_amp: bool = False):
    """
    一个通用的 REFCOCO 评测循环示例：
    - 使用 EMA 教师（若有）
    - 保持与训练相同的前向接口：训练时是 (pred, target, loss)，
      推理/评测时你的模型返回 (logits, aux) 或 (logits, attention) 皆可。
    - metric_fn: 你自己的评测函数，接收 (pred_mask, gt_mask) 或 (logits, gt_mask)

    返回 metric_fn 的聚合结果（如 {"miou": x, "oiou": y}）
    """
    eval_model = get_model_for_eval(model)
    eval_model.eval()

    # 兜底：把 eval 模型搬到与 dataloader 第一个 batch 一致的 device
    # 有的工程里 dataloader 的 pin_memory / non_blocking 会导致 batch 在 cuda，
    # 我们以第一批的图像设备为准。
    first_batch = True
    scaler = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast

    metric_fn.reset() if hasattr(metric_fn, "reset") else None

    for batch in dataloader:
        # 你自己的 batch 结构：假设 (image, text, target)
        image, text, target = batch

        # 第一次迭代时，确定 device，并把 eval_model 迁移到同一设备
        if first_batch:
            device_to_use = image.device if hasattr(image, "device") else device
            ensure_same_device(eval_model, device_to_use)
            first_batch = False

        image = image.to(device_to_use, non_blocking=True)
        text = text.to(device_to_use, non_blocking=True)
        target = target.to(device_to_use, non_blocking=True)

        with scaler():
            # 你的 eval 前向接口：在 segmenter.py 的 eval 分支里，返回 (logits, attention_or_none)
            # 这里我们只取 logits（pred）
            outputs = eval_model(image, text)  # eval 分支通常是两个返回值
            if isinstance(outputs, (tuple, list)):
                pred = outputs[0]  # B×1×H×W logits
            else:
                pred = outputs

        # 将 logits -> 概率/二值掩码，具体阈值由 metric 决定；也可以把 logits 直接交给 metric_fn
        # 示例：metric_fn.update(pred, target)
        if hasattr(metric_fn, "update"):
            metric_fn.update(pred, target)
        else:
            # 若你的 metric_fn 是纯函数式，就在此处聚合
            pass

    results = metric_fn.compute() if hasattr(metric_fn, "compute") else None
    return results
