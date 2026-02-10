import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import load_config, save_model

def train_model():
    cfg = load_config()
    
    # Load Data
    data_path = cfg['data']['output_path']
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['target'])
    y = df['target']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=cfg['data']['test_size'], 
        random_state=cfg['experiment']['random_seed']
    )

    # Pipeline: Scaling is crucial for SVM
    print(f"Training SVM with kernel: {cfg['model']['kernel']}...")
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=cfg['model']['kernel'],
            C=cfg['model']['C'],
            gamma=cfg['model']['gamma']
        )
    )

    model.fit(X_train, y_train)

    # Save
    save_model(model, cfg['model']['save_path'])

if __name__ == "__main__":
    train_model()