import numpy as np
import torch

class expFilter:
    def __init__(self,alpha):
        """
        指数滤波器\\
        alpha=0,完全不滤波,等于current\\
        alpha=1,全部等于last
        """
        self.alpha = alpha
        self.last = None

    def filter(self,current):
        if self.last is None:
            self.last = torch.zeros_like(current)
        filtered = current * (1-self.alpha) + self.last * self.alpha
        self.last = filtered
        return filtered
    
    def reset(self):
        self.last = None