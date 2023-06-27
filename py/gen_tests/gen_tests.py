from collections import OrderedDict
import sys

import torch
import torch.nn.functional as F
import numpy as np

g = torch.Generator()
g.manual_seed(42)

test_cases = OrderedDict()

pixel_values = torch.rand((3, 224, 224))
weight = torch.rand((1024, 3, 14, 14))
result = F.conv2d(pixel_values, weight, stride=14)
test_cases["clip.conv2d.pixel_values"] = pixel_values
test_cases["clip.conv2d.weight"] = weight
test_cases["clip.conv2d.result"] = result
torch.save(test_cases, sys.argv[1])
