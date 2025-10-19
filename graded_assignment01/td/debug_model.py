import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from modules.models import PHOSCnet

# Create a fresh model to see the architecture
model = PHOSCnet()

print("Model architecture:")
print("PHOS branch:")
for i, layer in enumerate(model.phos):
    print(f"  {i}: {layer}")

print("\nPHOC branch:")
for i, layer in enumerate(model.phoc):
    print(f"  {i}: {layer}")

# Check the final layers
print(f"\nFinal PHOS layer: {model.phos[-1]}")
print(f"Final PHOC layer: {model.phoc[-1]}")

# Test if there are activation functions
print(f"\nPHOS final layer weight range: {model.phos[-1].weight.min():.3f} to {model.phos[-1].weight.max():.3f}")
print(f"PHOC final layer weight range: {model.phoc[-1].weight.min():.3f} to {model.phoc[-1].weight.max():.3f}")