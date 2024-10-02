import torch
from torch import nn
class SampleConditionGMM(nn.Module):
    def __init__(self):
        super(SampleConditionGMM, self).__init__()

    def forward(self, labels):
        mean_distributions = torch.distributions.Uniform(low=0, high=255)
        std_distributions = torch.distributions.Uniform(low=0, high=30)
        classes = torch.unique(labels)
        class_means = mean_distributions.sample([classes.shape[0]])
        class_stds = std_distributions.sample([classes.shape[0]])
        images = torch.zeros(labels.shape)

        for i, c in enumerate(classes):
            samples = torch.normal(
                mean=class_means[i], std=class_stds[i], size=labels.shape
            )
            images[labels == c] = samples[labels == c]

        return images