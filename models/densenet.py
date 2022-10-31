from torchvision.models import densenet121
import torch.nn as nn


class DenseNet121(nn.Module):
    def __init__(self, num_classes, is_trained=True):
        """
        Init model architecture

        Parameters
        ----------
        num_classes: int
            number of classes
        is_trained: bool
            whether using pretrained model from ImageNet or not
        """
        super().__init__()

        # Load the DenseNet121 from ImageNet
        self.net = densenet121(pretrained=is_trained)

        # Get the input dimension of last layer
        kernel_count = self.net.classifier.in_features

        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        self.net.classifier = nn.Sequential(nn.Linear(kernel_count, num_classes), nn.Sigmoid())

    def forward(self, inputs):
        """
        Forward the netword with the inputs
        """
        return self.net(inputs)


if __name__ == '__main__':
    network = DenseNet121(1000)
    print(network)