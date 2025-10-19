import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
from modules.models import PHOSCnet
from modules.dataset import phosc_dataset
from torchvision.transforms import transforms

# Load trained model
model = PHOSCnet()
model.load_state_dict(torch.load('PHOSCnet_temporalpooling/epoch1.pt'))
model.eval()

print("Trained model outputs:")
# Test with a few samples from dataset
dataset = phosc_dataset(
    'dte2502_ga01_small/train.csv',
    'dte2502_ga01_small/train',
    transform=transforms.ToTensor()
)

for i in range(3):
    img, target, word = dataset[i]
    with torch.no_grad():
        output = model(img.unsqueeze(0))  # Add batch dimension

    print(f"\nSample {i}: '{word}'")
    print(f"  Target PHOC sum: {target[165:].sum():.1f}")  # PHOC part of target
    print(f"  Output PHOC sum: {output['phoc'][0].sum():.3f}")
    print(f"  Output PHOC range: [{output['phoc'][0].min():.3f}, {output['phoc'][0].max():.3f}]")
    print(f"  Output PHOC mean: {output['phoc'][0].mean():.3f}")

    # Check how many outputs are > 0.5 (reasonable threshold)
    phoc_binary = (output['phoc'][0] > 0.5).float()
    print(f"  Output PHOC > 0.5: {phoc_binary.sum().item()}/604")