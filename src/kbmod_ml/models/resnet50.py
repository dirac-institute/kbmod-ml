# ruff: noqa: D101, D102

# This example model is taken from the PyTorch CIFAR10 tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torch.optim as optim
from fibad.models.model_registry import fibad_model
from torchvision.models import resnet50

logger = logging.getLogger(__name__)


@fibad_model
class RESNET50(nn.Module):
    def __init__(self, model_config, shape):
        logger.info("This is an external model, not in FIBAD!!!")
        super().__init__()

        self.config = model_config

        self.model = resnet50(pretrained=False, num_classes=self.config["model"]["num_classes"])

        # Optimizer and criterion could be set directly, i.e. `self.optimizer = optim.SGD(...)`
        # but we define them as methods as a way to allow for more flexibility in the future.
        self.optimizer = self._optimizer()
        self.criterion = self._criterion()

    def forward(self, x):
        return self.model(x)

    def train_step(self, batch):
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
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def _criterion(self):
        return nn.CrossEntropyLoss()

    def _optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def save(self):
        torch.save(self.state_dict(), self.config.get("weights_filepath"))
