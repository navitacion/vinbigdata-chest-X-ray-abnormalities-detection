from dataclasses import dataclass, field
from typing import Dict

import torch


a = torch.randn(4, 2)
print(a)
print(torch.mean(a, 1))
print(torch.mean(a, 0))

b = torch.mean(a, 0)
print(b[1].item())