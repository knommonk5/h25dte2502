import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Add current directory to path
sys.path.append('..')

try:
    from modules.models import PHOSCnet
    import torch

    print("Model imports successfully!")

    # Test model creation
    model = PHOSCnet()
    print("Model created successfully!")

    # Test forward pass
    x = torch.randn(2, 3, 50, 250)
    output = model(x)
    print("Forward pass successful!")
    print(f"PHOS shape: {output['phos'].shape}")
    print(f"PHOC shape: {output['phoc'].shape}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()