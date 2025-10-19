import torch
import torch.nn as nn
from torch.nn import functional as F


class PHOSCLoss(nn.Module):
    def __init__(self, phos_w=1.0, phoc_w=10.0):
        #weights
        super().__init__()
        self.phos_w = phos_w
        self.phoc_w = phoc_w
        #PHOS: Mean Squared
        self.phos_loss_fn = nn.MSELoss()

    def forward(self, y: dict, targets: torch.Tensor):
        batch_size = targets.size(0)

        #split target vector
        phos_targets = targets[:, :165]  #PHOS spatial
        phoc_targets = targets[:, 165:]  #rest 604 dimensions PHOC character

        #predictions
        phos_output = y['phos']  #PHOS
        phoc_output = y['phoc']  #PHOC

        #mean squared error between predicted and PHOS vectors
        phos_loss = self.phos_w * self.phos_loss_fn(phos_output, phos_targets)

        pos_weight = torch.ones(604) * 5.0  #weight for positive examples

        #BCEWithLogitsLoss combines sigmoid + BCE
        phoc_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(phoc_output.device))
        phoc_loss = self.phoc_w * phoc_loss_fn(phoc_output, phoc_targets)

        #tot loss
        loss = phos_loss + phoc_loss

        #debugg
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss - PHOS: {phos_loss.item():.4f}, PHOC: {phoc_loss.item():.4f}")

        return loss