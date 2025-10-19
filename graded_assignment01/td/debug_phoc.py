import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils import generate_phoc_vector, generate_phos_vector

# Test PHOC generation
test_words = ['hello', 'test', 'a', 'world']
for word in test_words:
    phoc = generate_phoc_vector(word)
    phos = generate_phos_vector(word)
    print(f"Word: '{word}'")
    print(f"  PHOC length: {len(phoc)}, non-zero: {sum(phoc)}")
    print(f"  PHOS length: {len(phos)}, sum: {sum(phos)}")
    print(f"  PHOC vector sample: {phoc[:10]}")  # First 10 elements