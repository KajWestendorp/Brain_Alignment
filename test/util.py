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
        get_tvsd(subjects, normalized=ephys_normalized)
    elif dataset_name == "eeg":
        get_eeg(subjects)
    elif dataset_name == "fmri":
        get_fmri(subjects)
    
    pass


def get_dataloader(dataset_name, batch_size=128, num_workers=4):

    from sklearn.model_selection import train_test_split

    # Check which dataset and respond accordingly

    if dataset_name == 'THINGS':
        #  Here you want to somehow define your split for training and testing (and I guess also for using the EEG THINGS dataset or the ephys THINGS dataset
        THINGS_PATH = os.path.expanduser("~/Documents/BrainAlign_Data/things_imgsF")
        all_imgs_paths = get_image_paths(THINGS_PATH, 'things_imgsF')
        train_imgs_paths, test_imgs_paths = train_test_split(all_imgs_paths, test_size=0.2)

        # The below part should then go into the get_things_dataloader function which you then only call here
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = THINGS(root=THINGS_PATH, transform=transform, paths=train_imgs_paths)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        test_dataset = THINGS(root=THINGS_PATH, transform=transform, paths=test_imgs_paths)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    elif dataset_name == 'NSD':
        ... # do NSD dataloader preparation

    return train_dataloader, test_dataloader

def get_model(model_name):
    if model_name == 'alexnet':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        #Add more later
    return model

things_train_dataloader, things_test_dataloader = get_dataloader(dataset_name='THINGS')

model = get_model(model_name='alexnet')
device = torch.device('cuda')

feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
feature_extractor = feature_extractor.to(device)

############# Feature extraction part (this should also go into it's own function #############
def create_feature_extractor(model, return_nodes=False):
    with torch.no_grad():
        for item in tqdm.tqdm(things_train_dataloader, total=len(things_train_dataloader)):
        imgs, lbls = item # Usually a dataloader returns a batch of pairs containing images and labels. This you specify in the Dataset object that your dataloader gets, and you can also make it only return the images and not the labels. Then this line would only be imgs = item
        imgs = imgs.to(device)
        
        batch_activations = feature_extractor(imgs)




