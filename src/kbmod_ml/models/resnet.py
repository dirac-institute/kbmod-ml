# ruff: noqa: D101, D102

import logging

import torch
import torch.nn as nn
from hyrax.models.model_registry import hyrax_model
from torchvision.models import resnet18

logger = logging.getLogger(__name__)


@hyrax_model
class RESNET18(nn.Module):
    def __init__(self, config, data_sample=None):
        super().__init__()

        self.config = config

        self.model = resnet18(num_classes=2)

        # Modify the input channels to 1 (e.g., for grayscale images)
        self.model = self.modify_resnet_input_channels(self.model, num_channels=1)

    def forward(self, x):
        # if labels are passed to forward as part
        # of the infer step of training, just pass along the stamps.
        if isinstance(x, tuple) or isinstance(x, list):
            x, _ = x
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
        # inputs = batch["image"]
        # labels = batch["label"]

        self.optimizer.zero_grad()
        outputs = self(inputs)
        targets = torch.zeros((len(labels), 2))
        targets[labels == 0, 0] = 1
        targets[labels == 1, 1] = 1
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def modify_resnet_input_channels(self, model, num_channels):
        # Get the first convolutional layer
        first_conv_layer = model.conv1

        # Create a new convolutional layer with the desired number of input channels
        new_conv_layer = nn.Conv2d(
            in_channels=num_channels,
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=first_conv_layer.bias,
        )

        # Replace the first convolutional layer in the model
        model.conv1 = new_conv_layer

        return model

    @staticmethod
    def to_tensor(data):
        data = data["data"]
        classification = None
        if "classification" in data.keys():
            classification = data["classification"]
        return data["stamps"], classification
