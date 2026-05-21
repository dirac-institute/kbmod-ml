import logging
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from hyrax.models.model_registry import hyrax_model

logger = logging.getLogger(__name__)

# Module-level buffers for accumulating val predictions across epochs.

_val_probs_buffer = []
_val_labels_buffer = []
_val_buffer_lock = threading.Lock()


class SEBlock(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r), nn.ReLU(),
            nn.Linear(c // r, c), nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=[2, 3])
        return x * self.fc(w).unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.se = SEBlock(c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.conv(x)) + x)


def focal_loss(logits, targets, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction="none")
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean()


@hyrax_model
class KBModNet(nn.Module):
    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config
        in_channels = 3
        base = 32
        dropout = 0.3

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.GELU(),
        )
        self.layer1 = nn.Sequential(ResBlock(base), ResBlock(base))
        self.down1 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2), nn.GELU(),
        )
        self.layer2 = nn.Sequential(ResBlock(base * 2), ResBlock(base * 2))
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4), nn.GELU(),
        )
        self.layer3 = nn.Sequential(ResBlock(base * 4), ResBlock(base * 4))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base * 4, base * 2), nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(base * 2, 2),
        )

        self.criterion = focal_loss

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x, _ = x
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        return self.head(x)

    def infer_batch(self, batch):
        return self(batch)

    def validate_batch(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = focal_loss(outputs, labels)
        probs = F.softmax(outputs, dim=1)[:, 1].detach().cpu()
        with _val_buffer_lock:
            _val_probs_buffer.append(probs)
            _val_labels_buffer.append(labels.detach().cpu())
        return {"loss": loss.item()}

    def train_batch(self, batch):
        inputs, labels = batch
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = focal_loss(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        if not hasattr(self, "_epoch_losses"):
            self._epoch_losses = []
        self._epoch_losses.append(loss.item())
        return {"loss": loss.item()}

    def log_epoch_metrics(self):
        avg = sum(self._epoch_losses) / len(self._epoch_losses) if hasattr(self, "_epoch_losses") and self._epoch_losses else 0
        self._epoch_losses = []
        metrics = {"avg_train_loss": avg}

        with _val_buffer_lock:
            if _val_probs_buffer:
                from sklearn.metrics import roc_auc_score, average_precision_score
                probs  = torch.cat(_val_probs_buffer).numpy()
                labels = torch.cat(_val_labels_buffer).numpy()
                _val_probs_buffer.clear()
                _val_labels_buffer.clear()
                if len(set(labels)) > 1:
                    metrics["val_auc"] = roc_auc_score(labels, probs)
                    metrics["val_ap"]  = average_precision_score(labels, probs)

        self.final_epoch_metrics = metrics
        return metrics

    @staticmethod
    def prepare_inputs(data):
        data = data["data"]
        classification = None
        if "classification" in data.keys():
            classification = data["classification"]
        return data["stamps"], classification
