import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from modules.models import PHOSCnet
from modules.loss import PHOSCLoss

# Test loss calculation
model = PHOSCnet()
criterion = PHOSCLoss()

# Create realistic targets
batch_size = 2
x = torch.randn(batch_size, 3, 50, 250)

# Realistic PHOSC targets
phos_target = torch.randn(batch_size, 165) * 0.1  # Small values
phoc_target = torch.zeros(batch_size, 604)
# Set some random active features (like real PHOC)
for i in range(batch_size):
    active_indices = torch.randint(0, 604, (15,))  # ~15 active features
    phoc_target[i, active_indices] = 1.0

targets = torch.cat([phos_target, phoc_target], dim=1)

# Forward pass
outputs = model(x)
loss = criterion(outputs, targets)

print("Loss breakdown:")
print(f"Total loss: {loss.item():.4f}")

# Manual loss calculation to verify
phos_loss = criterion.phos_w * criterion.phos_loss_fn(outputs['phos'], targets[:, :165])
phoc_loss = criterion.phoc_w * criterion.phoc_loss_fn(outputs['phoc'], targets[:, 165:])
manual_total = phos_loss + phoc_loss

print(f"PHOS loss: {phos_loss.item():.4f}")
print(f"PHOC loss: {phoc_loss.item():.4f}")
print(f"Manual total: {manual_total.item():.4f}")
print(f"Matches: {torch.isclose(loss, manual_total)}")