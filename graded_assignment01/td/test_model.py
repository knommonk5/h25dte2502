import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
sys.path.append('.')

import torch
from modules.models import PHOSCnet

# Test model initialization
try:
    model = PHOSCnet()
    print("Model created successfully!")

    # Test forward pass
    x = torch.randn(2, 3, 50, 250)  # batch_size=2, channels=3, H=50, W=250
    output = model(x)

    print(f"Forward pass successful:")
    print(f"   PHOS output shape: {output['phos'].shape}")
    print(f"   PHOC output shape: {output['phoc'].shape}")
    print(f"   PHOS range: {output['phos'].min():.3f} to {output['phos'].max():.3f}")
    print(f"   PHOC range: {output['phoc'].min():.3f} to {output['phoc'].max():.3f}")

    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

except Exception as e:
    print(f"Model error: {e}")
    import traceback
    traceback.print_exc()