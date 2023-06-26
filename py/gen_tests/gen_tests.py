import torch
import numpy as np

g = torch.Generator()
g.manual_seed(42)

pixel_values = torch.rand((3, 224, 224))
print(f"{pixel_values}")