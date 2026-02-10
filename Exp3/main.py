from src.data_generator import generate_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("Starting SVM Experiment Pipeline...")
    print("="*30)
    
    # Step 1: Generate Data
    generate_data()
    print("-" * 30)
    
    # Step 2: Train Model
    train_model()
    print("-" * 30)
    
    # Step 3: Evaluate
    evaluate_model()
    print("="*30)
    print("Experiment Completed.")

if __name__ == "__main__":
    main()