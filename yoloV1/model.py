import torch
from torch import nn
from torch.nn import functional as F



class YOLO(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(YOLO, self).__init__()
        self.S = S # grid size
        