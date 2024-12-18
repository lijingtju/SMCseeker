import torch



class ExpMseLoss(torch.nn.Module):
    def __init__(self, t: float, alpha: float = 6, clamp_min: float = -1.8, clamp_max: float = 1.5) -> None:
        super(ExpMseLoss, self).__init__()
        self.t = t
        self.alpha = alpha
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, output, target):
        delta = torch.clamp(output - self.t, self.clamp_min, self.clamp_max)
        return torch.sqrt(torch.mean(self.alpha * torch.exp(delta) * (output - target) ** 2))