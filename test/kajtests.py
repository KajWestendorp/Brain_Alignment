import os
import yaml
import pandas as pd
from util import *
import similarity 
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import h5py
import numpy as np
import torch
from scipy.io import loadmat
import time  # Add this import for timing
from tqdm import tqdm  # For progress bars

if __name__ == '__main__':

    # ### Load config
    # config_filename = 'config.yaml'
    # with open(config_filename, 'r') as f:
    #     config = yaml.load(stream=f, Loader=yaml.FullLoader)

    # rows = []
    # cols = ['model', 'dataset', 'feature', "cv_split", 'metric_function', 'score_type', 'value']

    # # Get all the subjects that are already done
    # existing_subs = os.listdir(result_dir)
    # existing_subs = [sub for sub in existing_subs if sub.startswith('sub-')]


    # # for model in config['models']:

    # #     for modalities in config['modalities']:
    # #         pass

    # # 1. Select dataset
    # thingsF = "THINGS_normMUAF.mat"
    # print(get_neurodata("tvsd", thingsF))

    # # 2. Select model

    # # Select torch version of alexnet
    # model = models.alexnet(pretrained=True)
    # model_name = 'alexnet'

    # # 3. Extract features


    # # 4. Run metrics


    # # 5. Save results

    # corr = -0.1

    # rows.append([model_name, dataset, feature, cv_split, 'rsa', 'pearsons_correlation', corr])


    # df = pd.DataFrame(data=rows, columns=cols)

    # df.to_parquet('results.parquet')





    # ##################3333 To plot the data (but please do so in a different file)

    # df = pd.read_parquet('results.parquet')

    # import seaborn as sns
    # from matplotlib import pyplot as plt

    # g = sns.relplot(data=df, x='timepoint', y='value', hue='model', col='feature', row='dataset', kind='line')
    # plt.show()


    # Load the THINGS dataset
    things_train_dataloader, things_test_dataloader = get_dataloader(dataset_name='THINGS')

    # Load the model
    model = get_model(model_name='alexnet')
    device = torch.device('cuda')
    print(model)

    # Define the return nodes
    return_nodes = {
        'features.1': 'Relu1',
        'features.2': 'MaxPool1',
    }

    # Extract features for specified layers in return nodes
    # Extract features with PCA for dimensionality reduction
    features_train = extract_features(
        model, 
        things_train_dataloader, 
        device, 
        return_nodes,
        flatten_features=True,
        apply_pca=True,
        n_components=100  # Adjust based on how much variance you want to retain
    )
    features_test = extract_features(
        model, 
        things_test_dataloader, 
        device, 
        return_nodes,
        flatten_features=True,
        apply_pca=True,
        n_components=100
    )

    # Get neural data for V1
    # Fix the path using os.path.expanduser()
    filepath_tvsd = os.path.expanduser('~/Documents/BrainAlign_Data/THINGS_normMUAF.mat')
    V1, V4, IT = get_neurodata('tvsd', filepath_tvsd)

    # Check the shape of the extracted features and the keys
    print(features_train.keys())
    print(features_test.keys())
    print(features_train['Relu1'].shape)
    print(features_test['Relu1'].shape)
    print(features_train['MaxPool1'].shape)
    print(features_test['MaxPool1'].shape)


    # use the similarity repo to calculate the similarity between the features
    metrics = ["brainscore/cka"]  # 100% correct name
    
    results = {}
    timing_results = {}  # New dictionary to store timing information

    # Run the metrics
    for metric in metrics:
        try:
            # Print metric being used
            print(f"Trying metric: {metric}")
            measure = similarity.make(metric)
            
            # Get features from the model and move to CPU
            features = features_train['Relu1'].cpu()
            
            # Make sure V1 is a NumPy array
            if isinstance(V1, torch.Tensor):
                V1_numpy = V1.cpu().numpy()
            else:
                V1_numpy = np.asarray(V1)
            
            features_2d = features.numpy()  # Make sure features are numpy arrays
                
            print(f"Shapes - V1: {V1_numpy.shape}, features: {features_2d.shape}")
            
            # Apply the measure with timing
            print(f"Starting computation for {metric}...")
            start_time = time.time()
            
            with tqdm(total=1, desc=f"Computing {metric}") as pbar:
                score = measure(V1_numpy, features_2d)
                pbar.update(1)
                
            end_time = time.time()
            execution_time = end_time - start_time
            
            results[metric] = score
            timing_results[metric] = execution_time
            
            print(f"Score: {score}")
            print(f"Execution time for {metric}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
            
        except Exception as e:
            print(f"Error with metric {metric}: {str(e)}")
            
            # Try alternate metric name if first one fails
            try:
                alt_metric = "brainscore/cka"
                print(f"Trying alternative metric: {alt_metric}")
                measure = similarity.make(alt_metric)
                
                # Time the alternative approach
                start_time = time.time()
                
                # Make sure we're using the 2D version of features for the fallback
                if 'features_2d' not in locals():
                    features_2d = features.reshape(features.shape[0], -1).numpy()
                
                with tqdm(total=1, desc=f"Computing {alt_metric}") as pbar:
                    score = measure(V1_numpy, features_2d)
                    pbar.update(1)
                    
                end_time = time.time()
                execution_time = end_time - start_time
                
                results[alt_metric] = score
                timing_results[alt_metric] = execution_time
                
                print(f"Score with alternative metric: {score}")
                print(f"Execution time for {alt_metric}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
                
            except Exception as e2:
                print(f"Error with alternative approach: {str(e2)}")
                continue

    # Save the results to a file with timing information
    import json
    results_file = os.path.expanduser("~/Documents/BrainAlign_Data/alexnet_results.json")
    with open(results_file, "w") as file:
        # Convert results to JSON serializable format and include timing info
        json_results = {
            "scores": {k: float(v) if isinstance(v, np.number) else v for k, v in results.items()},
            "timing": {k: float(v) for k, v in timing_results.items()}
        }
        json.dump(json_results, file, indent=4)
        
    print(f"Results saved to {results_file}")
    print("Timing summary:")
    for metric, exec_time in timing_results.items():
        print(f"  {metric}: {exec_time:.2f} seconds ({exec_time/60:.2f} minutes)")