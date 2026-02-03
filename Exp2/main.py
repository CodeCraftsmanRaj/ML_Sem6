import config
from src.data_generator import generate_student_data
from src.plotting import save_tree_plot
from src.tree_utils import print_tree_structure, print_gini_indices, print_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    df = generate_student_data()
    
    # Save the generated dataset
    print(f"Saving dataset to {config.DATASET_PATH}...")
    df.to_csv(config.DATASET_PATH, index=False)
    
    X = df[config.FEATURES]
    y = df[config.TARGET]
    
    # 1. Split Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # 2. Split Validation set from Train set
    # Calculate relative validation size based on remaining data
    relative_val_size = config.VAL_SIZE / (1 - config.TEST_SIZE)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=config.RANDOM_STATE
    )
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    clf = DecisionTreeClassifier(**config.MODEL_PARAMS)
    clf.fit(X_train, y_train)
    
    train_accuracy = clf.score(X_train, y_train)
    val_accuracy = clf.score(X_val, y_val)
    test_accuracy = clf.score(X_test, y_test)
    
    print(f"Dataset generated with {config.N_SAMPLES} samples.")
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    print_tree_structure(clf, config.FEATURES)
    print_gini_indices(clf, config.FEATURES)
    print_feature_importance(clf, config.FEATURES)
    
    save_tree_plot(clf, config.FEATURES)

if __name__ == "__main__":
    main()