import os
import torch
from torch.utils.data import Dataset
from skimage import io
import pandas as pd
import numpy as np
from utils import generate_phoc_vector, generate_phos_vector


class phosc_dataset(Dataset):
    def __init__(self, csvfile, root_dir, transform=None, calc_phosc=True):
        self.root_dir = root_dir
        self.transform = transform

        #read CSV
        self.df = pd.read_csv(csvfile)

        #initialize lists for data
        image_names = []
        words = []
        phos_vectors = []
        phoc_vectors = []
        phosc_vectors = []

        #Process each row in CSV
        for _, row in self.df.iterrows():
            #handle different possible column names in CSV
            image_name = row['Image'] if 'Image' in row else row['image_name']
            word = row['Word'] if 'Word' in row else row['word']

            #generate phos vector: spatial character
            phos_vec = generate_phos_vector(word)
            #generate phoc vector: character n-gram
            phoc_vec = generate_phoc_vector(word)

            #combine phos and phoc into phosc vector
            phosc_vec = np.concatenate([phos_vec, phoc_vec])

            #store processed data
            image_names.append(image_name)
            words.append(word)
            phos_vectors.append(phos_vec)
            phoc_vectors.append(phoc_vec)
            phosc_vectors.append(phosc_vec)

        #create dataframe with processed info
        self.df_all = pd.DataFrame({
            'Image': image_names,
            'Word': words,
            'phos': phos_vectors,
            'phoc': phoc_vectors,
            'phosc': phosc_vectors
        })

        #debug information
        print(f"Dataset created with {len(self.df_all)} samples")
        if len(self.df_all) > 0:
            print(f"First image: {self.df_all.iloc[0, 0]}")
            print(f"First word: {self.df_all.iloc[0, 1]}")
            print(f"PHOSC vector length: {len(self.df_all.iloc[0, 4])}")

    def __getitem__(self, index):
        #create image path
        img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])

        try:
            #load image with scikit-image
            image = io.imread(img_path)

            #validate image load
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            #Check image
            if image.size == 0:
                raise ValueError(f"Empty image: {img_path}")

            #convert grayscale images to rgb
            if len(image.shape) == 2:  #Grayscale image
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 1:  #Single channel
                image = np.concatenate([image] * 3, axis=2)
            elif image.shape[2] == 4:  #rgba image
                image = image[:, :, :3]  #Remove alpha channel

            #PHOSC vector target
            phosc_vector = self.df_all.iloc[index, 4]

            #validate PHOSC vector dimensions
            if len(phosc_vector) != (165 + 604):
                raise ValueError(f"Invalid PHOSC vector length: {len(phosc_vector)}")

            #convert to pytorch tensor
            y = torch.tensor(phosc_vector, dtype=torch.float32)

            #apply image transformations if specified
            if self.transform:
                image = self.transform(image)

            #validation
            if torch.isnan(y).any() or torch.isinf(y).any():
                raise ValueError(f"Invalid target values in sample {index}")

            return image.float(), y.float(), self.df_all.iloc[index, 1]

        except Exception as e:
            #backup for corrupted samples, dummy data
            print(f"Error loading sample {index} ({img_path}): {e}")
            dummy_image = torch.randn(3, 50, 250)  #expected input size
            dummy_target = torch.zeros(165 + 604)  #PHOSC dimensions
            dummy_word = "ERROR"
            return dummy_image.float(), dummy_target.float(), dummy_word

    def __len__(self):
        #return total number of samples in dataset
        return len(self.df_all) if hasattr(self, 'df_all') else 0


if __name__ == '__main__':
    #test script to verify dataset
    from torchvision.transforms import transforms

    try:
        #create test dataset instance
        dataset = phosc_dataset('image_data/IAM_test_unseen.csv', '../image_data/IAM_test',
                                transform=transforms.ToTensor())
        print(f"Dataset length: {len(dataset)}")
        print(f"Sample item: {dataset[0]}")
    except Exception as e:
        print(f"Dataset test failed: {e}")