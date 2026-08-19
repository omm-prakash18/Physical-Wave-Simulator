"""
Learning rate and teacher forcing schedulers.
"""

import math
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup followed by cosine decay.

    During warmup: lr = base_lr * (step / warmup_steps)
    After warmup:  lr = base_lr * 0.5 * (1 + cos(π * progress))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            max(self.min_lr, base_lr * scale)
            for base_lr in self.base_lrs
        ]


class TeacherForcingScheduler:
    """
    Linearly decay teacher forcing ratio from 1.0 to 0.0 over training.

    This is "scheduled sampling" — the model starts with full teacher
    forcing and gradually learns to handle its own predictions.
    """

    def __init__(self, total_epochs: int, start_ratio: float = 1.0, end_ratio: float = 0.0):
        self.total_epochs = total_epochs
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

    def get_ratio(self, epoch: int) -> float:
        """Get teacher forcing ratio for current epoch."""
        progress = min(epoch / max(1, self.total_epochs - 1), 1.0)
        return self.start_ratio + (self.end_ratio - self.start_ratio) * progress
