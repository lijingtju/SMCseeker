import torch
def bmc_loss(pred, target, noise_var):
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var)
    return loss


class BMCLoss(torch.nn.Module):
    def __init__(self, init_noise_sigma, device):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device=device))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)