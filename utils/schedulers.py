# utils/schedulers.py
from torch.optim.lr_scheduler import LambdaLR

def _warmup_cosine_lambda(current_epoch, *, warmup_epochs, total_epochs, min_lr_ratio):
    """
    返回给 LambdaLR 的 LR 系数（相对 base_lr 的倍率），按 epoch 进行：
      - 前 warmup_epochs：线性从 0 -> 1
      - 之后：余弦从 1 -> min_lr_ratio
    """
    if current_epoch < warmup_epochs:
        # 线性热身：避免前期过大步长
        return float(current_epoch + 1) / float(max(1, warmup_epochs))
    # 进入余弦
    progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    # 余弦从 1 → min_lr_ratio
    import math
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

def build_warmup_cosine_scheduler(optimizer, *, total_epochs: int, warmup_epochs: int = 3, min_lr_ratio: float = 0.1):
    """
    total_epochs: 总训练轮数（用于归一化余弦进度）
    warmup_epochs: 热身轮数（1~5 通常足够）
    min_lr_ratio: 末尾最低学习率 = base_lr * min_lr_ratio（0.05~0.2 常用）
    """
    lr_lambda = lambda ep: _warmup_cosine_lambda(
        ep, warmup_epochs=warmup_epochs, total_epochs=total_epochs, min_lr_ratio=min_lr_ratio
    )
    return LambdaLR(optimizer, lr_lambda=lr_lambda)
