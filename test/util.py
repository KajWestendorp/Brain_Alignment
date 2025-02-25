################################## Imports

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

############################# Low-level Functions: Dataset
def get_things_dataloader(image_paths):
    """Dataloader for THINGS dataset"""

    # Define the transform
    transform = transforms.Compose([

        # Resize to 224x224
        transforms.Resize((224, 224)),  

        # Convert to tensor for PyTorch
        transforms.ToTensor(),  

        # Normalize the images
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Apparently this is the mean and std for ImageNet but gotta double check
    ])

    # Load the images and apply the transform
    # Collect the image paths and labels
    dataset = []
    labels = []
    class_names = sorted(os.listdir(image_paths))  # Sort folders to have consistent labels

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

    # Return the dataloader
    return DataLoader(dataset, batch_size=128, num_workers=4)
 
def get_NSD_dataloader(image_paths):
    pass

def get_tvsd(subject, normalized=True):
    pass

def get_eeg(subject):
    pass

def get_fmri(subject):
    pass




############################# High-level Functions

def get_neurodata(dataset_name, subjects, ephys_normalized=True):

    get_tvsd(subjects, normalized=ephys_normalized)
    pass


def get_dataloader(dataset_name):

    # Check which dataset and respond accordingly
    # Train dataloader and test dataloader, hardcode 
    pass

def get_model(model_name):
    pass