from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import numpy as np
import torch 
from scipy.io import loadmat
import os

#################### Get Image Paths ################

# function works for e.g. get_image_paths("/home/c13745859/BrainAlign_Data/things_imgs.mat", "test_imgs")

def get_image_paths(file_path, group_name, num_images=None):
    with h5py.File(file_path, "r") as file:
        group = file[group_name]
        things_path_refs = group['things_path']
        image_paths = []

        # Determine the number of images to process
        total_images = len(things_path_refs) if num_images is None else min(num_images, len(things_path_refs))

        for ref in things_path_refs[:total_images]:
            ref_obj = file[ref.item()]  
            path_str = ''.join(chr(c) for c in ref_obj[:].flatten())  
            path_str = path_str.replace("\\", "/")  # Ensure compatibility across OS

            # Prepend correct image directory
            full_path = os.path.join(file_path, path_str)
            image_paths.append(full_path)
        
    return image_paths

############################# Low-level Functions: Dataset
class THINGS(Dataset):
    def __init__(self, root, paths, transform=None, device='cuda'):
        self.paths = paths
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(os.path.join(path))

        if self.transform:
            img = self.transform(img)
        return img, 0., idx 
    

def get_tvsd(subject_file_path, normalized=True, device="cuda"):
    """Dataloader for neurodata from TVSD dataset, this function will return the data for a single subject separated into V1, V4, IT"""

    # Load the data from the h5 file
    with h5py.File(subject_file_path, "r") as file:
        data_dict = {key: torch.tensor(np.array(file[key]), dtype=torch.float32, device=device) for key in file.keys()}

    train_mua = data_dict.get("train_MUA")
    
    #Index the data into V1, V4, IT based on Papale paper
    V1 = train_mua[:, :512]
    V4 = train_mua[:, 512:768]
    IT = train_mua[:, 768:1024]
    
    return V1, V4, IT  # Return the data for V1, V4, IT


def get_eeg(subject):
    """Dataloader for EEG data"""
    pass


def get_fmri(subject):
    pass

