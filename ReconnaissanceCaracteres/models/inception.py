import torch.nn as nn
from torchvision import models
from torchvision.models.inception import Inception_V3_Weights, Inception3
from collections import namedtuple
from typing import Callable, Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class InceptionModel(Inception3):
    def __init__(self, num_classes=10):
        super(InceptionModel, self).__init__()
        self.inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)
