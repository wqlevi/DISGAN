import torch
import sys
from lightning.pytorch.utilities import model_summary as MS
import lightning as L
from importlib import import_module
from model.model_VGG16_IN_Local import Discriminator, Generator, dwt_UNet

model_type = sys.argv[1]
mod = import_module('model.model_VGG16_IN_Local')
model = getattr(mod, model_type)
class P_G(model, L.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.Tensor(1,1,64,64, 64)

G = P_G()
print(MS.ModelSummary(G, max_depth = -1))
