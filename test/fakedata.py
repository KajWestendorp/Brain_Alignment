import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

# Generate fake neural data (V1)
np.random.seed(42)
V1 = np.random.rand(100, 50)  # 100 samples, 50 features

# Generate fake AlexNet features (Relu1)
features_relu1 = np.random.rand(100, 50)  # 100 samples, 50 features

# Convert to torch tensors
V1_tensor = torch.tensor(V1, dtype=torch.float32)
features_relu1_tensor = torch.tensor(features_relu1, dtype=torch.float32)

# Calculate expected CKA
def linear_CKA(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    
    XXT = torch.mm(X, X.T)
    YYT = torch.mm(Y, Y.T)
    
    return (torch.mm(XXT.flatten().unsqueeze(0), YYT.flatten().unsqueeze(0).T).item() / 
            (torch.norm(XXT.flatten()) * torch.norm(YYT.flatten())))

cka_score = linear_CKA(V1_tensor, features_relu1_tensor)
print(f"Expected CKA score: {cka_score}")

# Calculate expected linear regression correlation
lr = LinearRegression()
lr.fit(features_relu1, V1)
V1_pred = lr.predict(features_relu1)
linear_regression_score = r2_score(V1, V1_pred)
print(f"Expected linear regression correlation score: {linear_regression_score}")

# Calculate expected PLS regression correlation
pls = PLSRegression(n_components=2)
pls.fit(features_relu1, V1)
V1_pls_pred = pls.predict(features_relu1)
plsr_score = r2_score(V1, V1_pls_pred)
print(f"Expected PLS regression correlation score: {plsr_score}")

# Print shapes for verification
print(f"Shapes - V1: {V1.shape}, features: {features_relu1.shape}")


################

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
    # Generate fake neural data (V1)
    np.random.seed(42)
    V1 = np.random.rand(100, 50)  # 100 samples, 50 features

    # Generate fake AlexNet features (Relu1)
    features_relu1 = np.random.rand(100, 50)  # 100 samples, 50 features

    # Convert to torch tensors
    V1_tensor = torch.tensor(V1, dtype=torch.float32)
    features_relu1_tensor = torch.tensor(features_relu1, dtype=torch.float32)

    # Convert to NumPy arrays for similarity library
    V1_numpy = V1_tensor.numpy()
    features_2d = features_relu1_tensor.numpy()

    # use the similarity repo to calculate the similarity between the features
    metrics = ["measure/brainscore/cka-kernel=(rbf-sigma={sigma})-hsic=gretton-distance=angular", 
               "measure/brainscore/linear_regression-pearsonr",
               "measure/brainscore/pls-pearsonr-cv=5folds",
               "measure/brainscore/rsa-rdm=correlation-compare=spearman",
               ]  # Ensure correct metric names
    
    results = {}
    timing_results = {}  # New dictionary to store timing information

    # Run the metrics
    for metric in metrics:
        try:
            # Print metric being used
            print(f"Trying metric: {metric}")
            measure = similarity.make(metric)
                
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
            
            print(f"Score for {metric}: {score}")
            print(f"Execution time for {metric}: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
            
        except Exception as e:
            print(f"Error with metric {metric}: {str(e)}")

    # Save the results to a file with timing information
    import json
    results_file = os.path.expanduser("~/Documents/BrainAlign_Data/alexnet_fakeresults.json")
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
        print(f"  {metric}: {exec_time:.2f} seconds ({execution_time/60:.2f} minutes)")