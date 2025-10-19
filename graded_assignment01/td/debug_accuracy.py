import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from modules.dataset import phosc_dataset
from modules.models import PHOSCnet
from utils import get_map_dict
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PHOSCnet()
model.load_state_dict(torch.load('PHOSCnet_temporalpooling/epoch1.pt'))
model.to(device)
model.eval()

# Test a single batch manually
dataset = phosc_dataset(
    'dte2502_ga01_small/test_seen.csv',
    'dte2502_ga01_small/test_seen',
    transform=transforms.ToTensor()
)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# Get word map
words = list(set(dataset.df_all['Word']))
word_map = get_map_dict(words)
print(f"Number of unique words: {len(words)}")

# Test one batch
for images, targets, true_words in loader:
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    # Combine PHOS and PHOC
    vectors = torch.cat((outputs['phos'], outputs['phoc']), dim=1).float()
    print(f"Output vectors shape: {vectors.shape}")
    print(f"Output vectors range: [{vectors.min():.3f}, {vectors.max():.3f}]")

    # Manual cosine similarity calculation
    word_matrix = torch.stack([torch.tensor(vec).float().to(device) for vec in word_map.values()])
    print(f"Word matrix shape: {word_matrix.shape}")

    # Normalize
    vectors = vectors / (vectors.norm(p=2, dim=1, keepdim=True) + 1e-8)
    word_matrix = word_matrix / (word_matrix.norm(p=2, dim=1, keepdim=True) + 1e-8)

    cosine_similarities = vectors @ word_matrix.T
    print(f"Cosine similarities shape: {cosine_similarities.shape}")
    print(f"Cosine similarities range: [{cosine_similarities.min():.3f}, {cosine_similarities.max():.3f}]")

    predicted_indices = cosine_similarities.argmax(dim=1)
    predicted_words = [list(word_map.keys())[idx] for idx in predicted_indices.cpu().numpy()]

    print(f"True words: {true_words}")
    print(f"Predicted words: {predicted_words}")
    print(f"Matches: {[true == pred for true, pred in zip(true_words, predicted_words)]}")

    break