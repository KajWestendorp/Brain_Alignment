import os
import yaml
import pandas as pd
from util import *
import similarity 

if __name__ == '__main__':

    ### Load config
    config_filename = 'config.yaml'
    with open(config_filename, 'r') as f:
        config = yaml.load(stream=f, Loader=yaml.FullLoader)

    rows = []
    cols = ['model', 'dataset', 'feature', "cv_split", 'metric_function', 'score_type', 'value']

    # Get all the subjects that are already done
    existing_subs = os.listdir(result_dir)
    existing_subs = [sub for sub in existing_subs if sub.startswith('sub-')]


    # for model in config['models']:

    #     for modalities in config['modalities']:
    #         pass

    # 1. Select dataset
    thingsF = "THINGS_normMUAF.mat"
    print(get_neurodata("tvsd", thingsF))

    # 2. Select model


    # 3. Extract features


    # 4. Run metrics


    # 5. Save results

    corr = -0.1

    rows.append([model_name, dataset, feature, cv_split, 'rsa', 'pearsons_correlation', corr])


    df = pd.DataFrame(data=rows, columns=cols)

    df.to_parquet('results.parquet')





    ##################3333 To plot the data (but please do so in a different file)

    df = pd.read_parquet('results.parquet')

    import seaborn as sns
    from matplotlib import pyplot as plt

    g = sns.relplot(data=df, x='timepoint', y='value', hue='model', col='feature', row='dataset', kind='line')
    plt.show()