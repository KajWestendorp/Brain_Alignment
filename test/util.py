############################# Low-level Functions: Dataset
def get_things_dataloader(image_paths):
    pass

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