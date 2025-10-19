import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
sys.path.append('.')

import torch
import torch.nn as nn
from modules.models import PHOSCnet
from modules.loss import PHOSCLoss

try:
    # Create model and loss
    model = PHOSCnet()
    criterion = PHOSCLoss()

    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 50, 250)

    # Create targets (PHOSC vectors: 165 PHOS + 604 PHOC = 769 total)
    phos_target = torch.randn(batch_size, 165)  # PHOS part
    phoc_target = torch.sigmoid(torch.randn(batch_size, 604))  # PHOC part (probabilities)
    targets = torch.cat([phos_target, phoc_target], dim=1)

    # Forward pass and loss calculation
    outputs = model(x)
    loss = criterion(outputs, targets)

    print("Loss calculation successful!")
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Loss requires_grad: {loss.requires_grad}")

    # Test backward pass
    loss.backward()
    print("Backward pass successful!")

except Exception as e:
    print(f"Loss error: {e}")
    import traceback
    traceback.print_exc()