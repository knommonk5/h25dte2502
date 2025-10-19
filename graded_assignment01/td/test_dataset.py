import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
sys.path.append('.')

from modules.dataset import phosc_dataset
from torchvision.transforms import transforms

# Test with small dataset first
try:
    dataset = phosc_dataset(
        'dte2502_ga01_small/train.csv',
        'dte2502_ga01_small/train',
        transform=transforms.ToTensor()
    )
    print("Dataset created successfully!")
    print(f"Dataset length: {len(dataset)}")

    # Test single sample
    img, target, word = dataset[0]
    print(f"Single sample loaded:")
    print(f"   Image shape: {img.shape}")
    print(f"   Target shape: {target.shape}")
    print(f"   Word: {word}")
    print(f"   Target min/max: {target.min():.3f}, {target.max():.3f}")

except Exception as e:
    print(f"Dataset error: {e}")
    import traceback
    traceback.print_exc()