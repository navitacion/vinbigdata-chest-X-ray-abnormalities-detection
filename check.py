from dataclasses import dataclass, field
from typing import Dict

import torch
import os
import pandas as pd

data_dir = './input/resize_1024'
image_id = '002a34c58c5b758217ed1f584ccbcfe9'

test_image_info = pd.read_csv(os.path.join(data_dir, 'test_image_info.csv'))
row = test_image_info[test_image_info['image_id'] == image_id]
org_height, org_width = row['height'].values[0], row['width'].values[0]

print(org_width, org_height)