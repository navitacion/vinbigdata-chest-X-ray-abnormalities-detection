from dataclasses import dataclass, field
from typing import Dict

import torch
import os
import pandas as pd


# output_dir = './detectron2_output/exp01'
# # Metrics
# metrics_df = pd.read_json(os.path.join(output_dir, 'metrics.json'), orient="records", lines=True)
# mdf = metrics_df.sort_values("iteration")
#
# mdf3 = mdf[~mdf["bbox/AP75"].isna()].reset_index(drop=True)
#
# res = []
#
# for i in range(len(mdf3)):
#     row = mdf3.iloc[i]
#     res.append(row["bbox/AP75"] / 100.)
#
# print(res)
#


from torch import nn
from pytorch_lightning import metrics
z = torch.randn((4, 1)).float()
t = torch.tensor([0, 1, 0, 1]).float()

# acc_fn = metrics.Accuracy()
#
# z = torch.sigmoid(z)
#
# acc = acc_fn(z, t)

print(z)
print(t)
print(z.size())
print(t.size())
# print(acc)

z = z.reshape(t.size())
print(z.size())