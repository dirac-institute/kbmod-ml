# ruff: noqa: D101, D102

import logging

import torch
import torch.nn as nn
from hyrax.models.model_registry import hyrax_model
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class SEBlock(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c//r), nn.ReLU(),
            nn.Linear(c//r, c), nn.Sigmoid()
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
        self.se  = SEBlock(c)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.se(self.conv(x)) + x)

@hyrax_model
class KBModNet(nn.Module):
    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config
        in_channels = 3
        base=32
        dropout=0.3
        self.stem   = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.GELU(),
        )
        self.layer1 = nn.Sequential(ResBlock(base),   ResBlock(base))
        self.down1  = nn.Sequential(
            nn.Conv2d(base, base*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base*2), nn.GELU(),
        )
        self.layer2 = nn.Sequential(ResBlock(base*2), ResBlock(base*2))
        self.down2  = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base*4), nn.GELU(),
        )
        self.layer3 = nn.Sequential(ResBlock(base*4), ResBlock(base*4))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base*4, base*2), nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(base*2, 2),
        )
        self.criterion = focal_loss
    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
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
        
        self.optimizer.zero_grad()
        outputs = self(inputs)
        targets = torch.zeros((len(labels), 2))
        targets[labels == 0, 0] = 1
        targets[labels == 1, 1] = 1
        loss = focal_loss(outputs, targets)
        # loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": loss.item()}
    
    def train_batch(self, batch):
        """This function contains the logic for a single training step. i.e. the
        contents of the inner loop of a ML training process.

        Parameters
        ----------
        batch : tuple
            A tuple containing the inputs and labels for the current batch.

        Returns
        -------
        Current loss value
            The loss value for the current batch.
        """
        inputs, labels = batch

        self.optimizer.zero_grad()
        outputs = self(inputs)
        targets = torch.zeros((len(labels), 2))
        targets[labels == 0, 0] = 1
        targets[labels == 1, 1] = 1
        loss = focal_loss(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": loss.item()}
    
    @staticmethod
    def prepare_inputs(data):
        data = data["data"]
        classification = None
        if "classification" in data.keys():
            classification = data["classification"]
        return data["stamps"], classification

##changed loss function
def focal_loss(logits, targets, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction="none")
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean()


# def train(model, train_loader, val_loader, epochs=50, lr=3e-4, device="cuda"):
#     model    = model.to(device)
#     opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
#     history  = {"train_loss": [], "val_loss": [], "val_auc": [], "val_ap": []}
#     best_auc = 0.0

#     for epoch in range(epochs):
#         model.train()
#         losses = []
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)
#             opt.zero_grad()
#             loss = focal_loss(model(x), y)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()
#             losses.append(loss.item())
#         sched.step()

#         model.eval()
#         val_losses, all_probs, all_labels = [], [], []
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x, y = x.to(device), y.to(device)
#                 val_losses.append(focal_loss(model(x), y).item())
#                 probs = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
#                 all_probs.extend(probs)
#                 all_labels.extend(y.cpu().numpy())

#         auc = roc_auc_score(all_labels, all_probs)
#         ap  = average_precision_score(all_labels, all_probs)
#         history["train_loss"].append(np.mean(losses))
#         history["val_loss"].append(np.mean(val_losses))
#         history["val_auc"].append(auc)
#         history["val_ap"].append(ap)

#         print(f"Epoch {epoch+1:3d}/{epochs}  train={np.mean(losses):.4f}  "
#               f"val={np.mean(val_losses):.4f}  AUC={auc:.4f}  AP={ap:.4f}")

#         if auc > best_auc:
#             best_auc = auc
#             torch.save(model.state_dict(), SAVE_DIR + "best_kbmodnet_5_7sigma.pt")
#             print(f"  ↑ saved (AUC={auc:.4f})")

#     return history