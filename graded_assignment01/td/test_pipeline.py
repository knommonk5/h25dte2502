import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
sys.path.append('.')

import torch
from torch.utils.data import DataLoader
from modules.dataset import phosc_dataset
from modules.models import PHOSCnet
from modules.loss import PHOSCLoss
from torchvision.transforms import transforms

def test_full_pipeline():
    try:
        # 1. Dataset
        dataset = phosc_dataset(
            'dte2502_ga01_small/train.csv',
            'dte2502_ga01_small/train',
            transform=transforms.ToTensor()
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 2. Model
        model = PHOSCnet()
        criterion = PHOSCLoss()

        # 3. Single batch training step
        for batch_idx, (images, targets, words) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Images: {images.shape}")
            print(f"  Targets: {targets.shape}")
            print(f"  Words: {words}")

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            print(f"  Loss: {loss.item():.4f}")

            # Backward pass
            loss.backward()
            print("  Backward pass successful!")

            # Only test one batch
            break

        print("Full pipeline test passed!")

    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_pipeline()