import torch
from torch import nn
import numpy as np

from net.YOLOv4 import YOLOv4

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = YOLOv4().to(device)

x = torch.randn((1,1,32,32,32), device=device)
out = model(x)
x2 = out[1]
print(torch.equal(x,x2))
print([f.shape for f in out[0]], out[1].shape) # (list_of_3_pred_tensors, x)