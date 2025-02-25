################################## Imports

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import numpy as np
import torch 

############################# Low-level Functions: Dataset

def get_things_dataloader(image_paths, device="cuda"):
    """Dataloader for THINGS dataset"""

    # Define the transform (matching AlexNet's ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor for PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet (doulbe check this)
    ])

    # Collect the image paths and labels
    dataset = []
    labels = []
    class_names = sorted(os.listdir(image_paths))  

    # Traverse the directories to load images and assign labels based on folder names
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(image_paths, class_name)
        
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'): 
                    img_path = os.path.join(class_dir, img_name)
                    image = Image.open(img_path).convert("RGB")
                    image = transform(image)
                    dataset.append(image)
                    labels.append(label)

    # Send dataset and labels to the CUDA device
    dataset = [image.to(device) for image in dataset]  
    labels = torch.tensor(labels).to(device)  

    # Return the dataloader
    return DataLoader(list(zip(dataset, labels)), batch_size=128, num_workers=4)

 
def get_NSD_dataloader(image_paths, device="cuda"):
    """Dataloader for NSD dataset (perhaps could be the same as the THINGS dataset, depending on t he structure)"""
    
    # Define the transform (matching AlexNet's ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor for PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet (doulbe check this)
    ])

    # Collect the image paths and labels
    dataset = []
    labels = []
    class_names = sorted(os.listdir(image_paths))  

    # Traverse the directories to load images and assign labels based on folder names
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(image_paths, class_name)
        
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'): 
                    img_path = os.path.join(class_dir, img_name)
                    image = Image.open(img_path).convert("RGB")
                    image = transform(image)
                    dataset.append(image)
                    labels.append(label)

    # Send dataset and labels to the CUDA device
    dataset = [image.to(device) for image in dataset]  
    labels = torch.tensor(labels).to(device)  

    # Return the dataloader
    return DataLoader(list(zip(dataset, labels)), batch_size=128, num_workers=4)

def get_tvsd(subject_file_path, normalized=True, device="cuda"):
    """Dataloader for neurodata from TVSD dataset"""

    # Load the data from the h5 file
    with h5py.File(subject_file_path, "r") as file:
        data_dict = {key: torch.tensor(np.array(file[key]), dtype=torch.float32, device=device) for key in file.keys()}

    train_mua = data_dict.get("train_MUA")
    
    #Index the data into V1, V4, IT based on Papale paper
    V1 = train_mua[:, :512]
    V4 = train_mua[:, 512:768]
    IT = train_mua[:, 768:1024]
    
    return V1, V4, IT  # Each is a tensor now



def get_eeg(subject):
    pass

def get_fmri(subject):
    pass




############################# High-level Functions

def get_neurodata(dataset_name, subjects, ephys_normalized=True):

    if dataset_name == "tvsd":
        get_tvsd(subjects, normalized=ephys_normalized)
    elif dataset_name == "eeg":
        get_eeg(subjects)
    elif dataset_name == "fmri":
        get_fmri(subjects)
    
    pass


def get_dataloader(dataset_name):

    # Check which dataset and respond accordingly
    # Train dataloader and test dataloader, hardcode for now, how would i do this?
    if dataset_name == "THINGS":
        get_things_dataloader()
    elif dataset_name == "NSD":
        get_NSD_dataloader()

def get_model(model_name):
    pass