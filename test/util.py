################################## Imports

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import numpy as np
import torch 

###################### Helper Functions

def get_image_paths(file_path, group_name):
    with h5py.File(file_path, "r") as file:
        # Get the specified group
        group = file[group_name]
        
        # Dereference the image path references
        things_path_refs = group['things_path']
        image_paths = []
        
        # Dereference each reference to retrieve the actual image paths
        for ref in things_path_refs:
            ref_obj = file[ref.item()]  # Dereference the reference properly
            # Convert the uint16 array to a string
            path_str = ''.join(chr(c) for c in ref_obj[:].flatten())  # Flatten to 1D and convert
            image_paths.append(path_str)  # Append the string
            
        # Return the list of image paths
        return image_paths
file_path = os.path.expanduser("~/Documents/BrainAlign_Data/things_imgsF.mat")
all_imgs_paths = get_image_paths(file_path, 'things_imgsF')

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
        img = Image.open(os.path.join(self.root, path))

        if self.transform:
            img = self.transform(img)
        return img, 0., idx 
    
class NSD(Dataset):
    def __init__(self, root, paths, transform=None, device='cuda'):
        self.paths = paths
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(os.path.join(self.root, path))

        if self.transform:
            img = self.transform(img)
        return img, 0., idx

def get_things_dataloader(transform, THINGS_PATH,train_imgs_paths, test_imgs_paths, batch_size=128, num_workers=4):
    """Function to get the dataloader for the THINGS dataset"""
    
    train_dataset = THINGS(root=THINGS_PATH, transform=transform, paths=train_imgs_paths)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    test_dataset = THINGS(root=THINGS_PATH, transform=transform, paths=test_imgs_paths)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, test_dataloader

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
    pass

def get_fmri(subject):
    pass



############################# High-level Functions

def get_neurodata(dataset_name, subjects, ephys_normalized=True):

    if dataset_name == "tvsd":
        return get_tvsd(subjects, normalized=ephys_normalized)
    elif dataset_name == "eeg":
        return get_eeg(subjects)
    elif dataset_name == "fmri":
        return get_fmri(subjects)

def get_dataloader(dataset_name, batch_size=128, num_workers=4):
    """Function to get the dataloader for the specified dataset
    
    Args:
        dataset_name (str): The name of the dataset to get the dataloader for
        batch_size (int): The batch size for the dataloader
        num_workers (int): The number of workers for the dataloader
        
    Returns:
        train_dataloader, test_dataloader: The dataloaders for the training and testing sets
    """
    from sklearn.model_selection import train_test_split

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Check which dataset and respond accordingly
    if dataset_name == 'THINGS':
        #  Here you want to somehow define your split for training and testing (and I guess also for using the EEG THINGS dataset or the ephys THINGS dataset
        # I did so using the train_tset_split function from sklearn Idk if that's the best way to do it
        THINGS_PATH = os.path.expanduser("~/Documents/BrainAlign_Data/things_imgsF")
        all_imgs_paths = get_image_paths(THINGS_PATH, 'things_imgsF')
        train_imgs_paths, test_imgs_paths = train_test_split(all_imgs_paths, test_size=0.2)

        # The below part should then go into the get_things_dataloader function which you then only call here
        train_dataloader, test_dataloader = get_things_dataloader(transform,THINGS_PATH, train_imgs_paths, test_imgs_paths, batch_size=batch_size, num_workers=num_workers)

        return train_dataloader, test_dataloader
        
    elif dataset_name == 'NSD':
        ... # do NSD dataloader preparation

    return train_dataloader, test_dataloader

def get_model(model_name):
    if model_name == 'alexnet':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        #Add more later
    return model


################################## Feature Extraction functions
import torch
import tqdm
from torchvision.models.feature_extraction import create_feature_extractor

def extract_features(model, dataloader, device, return_nodes=None):
    """
    Extracts features from specific layers of a pre-trained model.

    Args:
        model (torch.nn.Module): Pretrained model for feature extraction.
        dataloader (DataLoader): DataLoader to fetch images in batches.
        device (torch.device): Device to run inference on (CPU/GPU).
        return_nodes (dict, optional): Dictionary mapping layer names to output names.
    
    Returns:
        dict: Dictionary of extracted features from specified layers.
    """
    # Apply return_nodes if specified
    if return_nodes:
        model = create_feature_extractor(model, return_nodes=return_nodes)

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval() 
    
    # Initialize dictionary to store features for each layer with the key as the layer name
    all_features = {key: [] for key in return_nodes.values()}  


    # Iterate over the dataloader to extract features
    with torch.no_grad():
        for item in tqdm.tqdm(dataloader, total=len(dataloader)):
            imgs, _ = item 
            imgs = imgs.to(device)
            
            batch_activations = model(imgs) 
            for key, activation in batch_activations.items():
                all_features[key].append(activation.cpu())  
    # Concatenate all batch features for each layer
    all_features = {key: torch.cat(features, dim=0) for key, features in all_features.items()}
    
    return all_features


###################### Feature Extraction Pipeline

things_train_dataloader, things_test_dataloader = get_dataloader(dataset_name='THINGS')

model = get_model(model_name='alexnet')
device = torch.device('cuda')

return_nodes = {
    'features.0': 'conv1', 
    'features.3': 'conv2', 
    'features.6': 'conv3', 
    'features.8': 'conv4', 
    'features.10': 'conv5'
}

# Extract features for specified layers in return nodes
features_train = extract_features(model, things_train_dataloader, device, return_nodes)
features_test = extract_features(model, things_test_dataloader, device, return_nodes)
