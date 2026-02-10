import pandas as pd
from sklearn.datasets import make_classification
import os
from src.utils import load_config

def generate_data():
    cfg = load_config()
    data_cfg = cfg['data']

    print("Generating synthetic data...")
    X, y = make_classification(
        n_samples=data_cfg['n_samples'],
        n_features=data_cfg['n_features'],
        n_informative=data_cfg['n_informative'],
        n_redundant=data_cfg['n_redundant'],
        n_clusters_per_class=data_cfg['n_clusters_per_class'],
        class_sep=data_cfg['class_sep'],
        random_state=cfg['experiment']['random_seed']
    )

    # Create DataFrame
    cols = [f"feature_{i+1}" for i in range(data_cfg['n_features'])]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y

    # Save
    os.makedirs(os.path.dirname(data_cfg['output_path']), exist_ok=True)
    df.to_csv(data_cfg['output_path'], index=False)
    print(f"Data generated and saved to {data_cfg['output_path']}")

if __name__ == "__main__":
    generate_data()